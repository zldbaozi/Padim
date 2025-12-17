import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

# 添加相对导入
try:
    from .feature_extractor import FeatureExtractor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.feature_extractor import FeatureExtractor

class PaDiMDetector:
    def __init__(self, trainer=None, model_dir=None):
        self.indices = None
        
        if trainer is not None:
            self.device = trainer.device
            self.feature_extractor = trainer.feature_extractor
            # 从 trainer 获取参数并转为 Tensor
            self.means = torch.from_numpy(trainer.means).to(self.device).float()
            self.covs = torch.from_numpy(trainer.covs).to(self.device).float()
            if hasattr(trainer, 'selected_indices'):
                self.indices = trainer.selected_indices
        elif model_dir is not None:
            self.load_model(model_dir)
        else:
            raise ValueError("必须提供trainer或model_dir")
        
        # 预计算逆协方差矩阵 (并保持在 GPU 上)
        self._precompute_inv_covs()
    
    def load_model(self, model_dir):
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.device = config.get('device', 'cuda')
        
        # 1. 加载模型
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # 2. 加载统计参数 (Numpy -> GPU Tensor)
        print("正在加载统计参数到 GPU...")
        means_np = np.load(os.path.join(model_dir, 'means.npy'))
        covs_np = np.load(os.path.join(model_dir, 'covs.npy'))
        
        self.means = torch.from_numpy(means_np).to(self.device).float()
        self.covs = torch.from_numpy(covs_np).to(self.device).float()
        
        # 3. 加载随机索引 (替代投影器)
        index_path = os.path.join(model_dir, 'selected_indices.pt')
        if os.path.exists(index_path):
            self.indices = torch.load(index_path).to(self.device)
            print(f"✅ 加载随机通道索引: {len(self.indices)} 维")
        else:
            self.indices = None
            print("⚠️ 未找到随机索引，使用全量特征")

    def _precompute_inv_covs(self):
        """在 GPU 上预计算协方差矩阵的逆"""
        print("正在 GPU 上预计算逆协方差矩阵...")
        # covs shape: [H, W, C, C]
        # 使用 torch.linalg.pinv 或 inverse 计算伪逆
        # 为了数值稳定性，通常加一个微小的 identity 已经在训练时做过了
        
        # 将 [H, W, C, C] 展平为 [H*W, C, C] 进行批量求逆
        H, W, C, _ = self.covs.shape
        covs_flat = self.covs.view(-1, C, C)
        
        # 批量求逆 (GPU 加速)
        inv_covs_flat = torch.linalg.pinv(covs_flat)
        
        # 恢复形状
        self.inv_covs = inv_covs_flat.view(H, W, C, C)
        print("✅ 逆矩阵计算完成")
    
    def predict(self, images):
        """
        全 GPU 推理流程
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            # 1. 预处理：先缩放到 112×112，再填充到 132×132
            images = F.interpolate(images, size=(112, 112), mode='bilinear', align_corners=False)
            
            
            # 2. 提取特征 [B, H, W, C_total]
            features = self._extract_multiscale_features(images)
            
            # 3. 极速降维 (GPU 切片)
            if self.indices is not None:
                # index_select 在 dim=3 (Channel) 上操作
                features = torch.index_select(features, 3, self.indices)
            
            # 4. 计算马氏距离 (全 GPU 矩阵运算，无循环)
            # features: [B, H, W, C]
            # means:    [H, W, C]
            # inv_covs: [H, W, C, C]
            
            # (x - mu)
            # 广播机制: [B, H, W, C] - [1, H, W, C]
            diff = features - self.means.unsqueeze(0) 
            
            # 计算 (x-mu) @ inv_cov @ (x-mu)^T
            # 步骤 A: temp = diff @ inv_cov
            # 使用 einsum 进行批量矩阵乘法
            # b: batch, h: height, w: width, i: c_in, j: c_out
            # diff: bhwi, inv_covs: hwij -> temp: bhwj
            temp = torch.einsum('bhwi,hwij->bhwj', diff, self.inv_covs)
            
            # 步骤 B: dist = temp * diff (求和)
            # bhwj, bhwj -> bhw
            mahalanobis_dist = torch.einsum('bhwj,bhwj->bhw', temp, diff)
            
            # 开根号
            mahalanobis_dist = torch.sqrt(torch.abs(mahalanobis_dist))
            
            # 5. 结果处理
            # 此时数据还在 GPU 上，如果需要返回 numpy，再转 cpu
            anomaly_maps = mahalanobis_dist.cpu().numpy()
            
            # 计算图像级分数 (取最大值)
            scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
        
        return anomaly_maps, scores
    
    def _extract_multiscale_features(self, images):
        features_list = self.feature_extractor(images)
        # 强制设置目标尺寸为 28*28
        target_size = (28, 28)
        
        resized_features = []
        for features in features_list:
            # 只有尺寸不对时才插值
            if features.shape[2:] != target_size:
                features = F.interpolate(features, size=target_size, 
                                       mode='bilinear', align_corners=False)
            resized_features.append(features)
        
        # [B, C, H, W]
        concatenated = torch.cat(resized_features, dim=1)
        # 转为 [B, H, W, C] 以便后续计算
        concatenated = concatenated.permute(0, 2, 3, 1)
        return concatenated
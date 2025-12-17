import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import time
import sys

# 添加相对导入
try:
    from .feature_extractor import FeatureExtractor
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.feature_extractor import FeatureExtractor

class PaDiMTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_extractor = FeatureExtractor().to(device)
        self.feature_extractor.eval()
        
        self.means = None
        self.covs = None
        self.image_size = None
        self.projector = None
        
    def fit(self, dataloader, reduce_dims=100):
    #"""使用正常图像训练PaDiM模型 (GPU加速版)"""
        
        print("开始提取正常图像的特征...")
        all_features = []

        # 第一步：提取所有正常图像的特征
        for batch in tqdm(dataloader, desc="提取特征"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            with torch.no_grad():  # 加上 no_grad 节省显存
                features = self._extract_multiscale_features(images)
            # 暂时保持在 GPU 上，不要 .cpu().numpy()，为了后续 GPU 切片加速
            all_features.append(features)

        # 合并所有特征 [N, H, W, C] (在 GPU 上合并)
        all_features = torch.cat(all_features, dim=0)
        print(f"特征形状: {all_features.shape}")

        # ==========================================
        # 核心修改：使用随机通道选择替代矩阵投影
        # ==========================================
        self.selected_indices = None

        if reduce_dims and reduce_dims < all_features.shape[-1]:
            total_dims = all_features.shape[-1]
            print(f"⚡ [极速推理优化] 使用随机通道选择: {total_dims} -> {reduce_dims}")

            # 1. 生成随机索引 (只做一次)
            # 随机选 reduce_dims 个不重复的通道
            self.selected_indices = torch.randperm(total_dims)[:reduce_dims].to(self.device)

            # 2. 立即切片 (零计算量)
            # [N, H, W, C] -> 在最后一个维度 C 上切片
            all_features = torch.index_select(all_features, -1, self.selected_indices)

            print(f"✅ 通道选择完成，当前形状: {all_features.shape}")

        # 转回 CPU 进行统计计算 (因为协方差矩阵计算在 CPU numpy 上可能更稳定，或者你可以尝试用 torch.cov 在 GPU 上算)
        print("将特征转移至 CPU 进行统计计算...")
        all_features = all_features.cpu().numpy()

        # 第三步：为每个位置计算多元高斯参数
        # ... (这部分保持不变，计算 means 和 covs) ...
        print("=" * 60)
        print("开始计算统计参数...")
        N, H, W, C = all_features.shape

        self.means = np.zeros((H, W, C))
        self.covs = np.zeros((H, W, C, C))
        
        print(f"需要处理 {H}×{W} = {H*W} 个空间位置")
        print(f"每个位置计算 {C}×{C} 的协方差矩阵")
        print(f"总计算量: {H*W} 个位置 × {C}×{C} 矩阵")
        print("=" * 60)
        
        # 添加详细的进度显示
        total_positions = H * W
        start_time = time.time()
        
        # 使用tqdm显示行进度
        for i in tqdm(range(H), desc="处理行进度"):
            row_start_time = time.time()
            
            for j in range(W):
                # 获取所有图像在位置(i,j)的特征
                patch_features = all_features[:, i, j, :]  # [N, C]
                
                # 计算均值
                self.means[i, j] = np.mean(patch_features, axis=0)
                
                # 计算协方差（添加正则化确保数值稳定性）
                self.covs[i, j] = np.cov(patch_features.T) + 0.01 * np.eye(C)
            
            # 每行完成后显示统计信息
            current_time = time.time()
            row_elapsed = current_time - row_start_time
            total_elapsed = current_time - start_time
            
            # 计算进度和预估时间
            completed_rows = i + 1
            completed_positions = completed_rows * W
            progress_percent = completed_positions / total_positions * 100
            
            # 计算速度
            positions_per_second = completed_positions / total_elapsed
            
            # 计算剩余时间
            remaining_positions = total_positions - completed_positions
            if positions_per_second > 0:
                estimated_remaining = remaining_positions / positions_per_second
            else:
                estimated_remaining = 0
            
            # 每2行或每分钟更新一次详细信息
            if (i % 2 == 0) or (i + 1 == H) or (total_elapsed > 60 and current_time - start_time > 60):
                print(f"📊 进度: {completed_rows}/{H} 行 | "
                      f"{completed_positions}/{total_positions} 位置 ({progress_percent:.1f}%)")
                print(f"⏱️  本行耗时: {row_elapsed:.1f}s | "
                      f"总耗时: {total_elapsed:.1f}s | "
                      f"预计剩余: {estimated_remaining:.1f}s")
                print(f"🚀 处理速度: {positions_per_second:.2f} 位置/秒")
                print("-" * 50)
        
        # 训练完成统计
        total_time = time.time() - start_time
        print("=" * 60)
        print("✅ 统计计算完成!")
        print(f"✅ 总耗时: {total_time:.2f} 秒")
        print(f"✅ 平均速度: {total_positions/total_time:.2f} 位置/秒")
        print(f"✅ 处理了 {total_positions} 个位置")
        print("=" * 60)
        
        self.image_size = dataloader.dataset[0][0].shape[-1] if hasattr(dataloader, 'dataset') else 224
        print("PaDiM训练完成!")
    
    def _extract_multiscale_features(self, images):
        """
        提取并融合多尺度特征
        """
        features_list = self.feature_extractor(images)
        
        # 将特征图调整到相同尺寸并拼接
        target_size = features_list[0].shape[2:]  # 以最大的特征图为目标
        
        resized_features = []
        for features in features_list:
            # 使用双线性插值调整特征图尺寸
            if features.shape[2:] != target_size:
                features = F.interpolate(features, size=target_size, 
                                       mode='bilinear', align_corners=False)
            resized_features.append(features)
        
        # 在通道维度拼接 [B, C1+C2+C3, H, W]
        concatenated = torch.cat(resized_features, dim=1)
        
        # 调整维度为 [B, H, W, C] 便于后续处理
        concatenated = concatenated.permute(0, 2, 3, 1)
        
        return concatenated
    
    def save_model(self, output_dir):
        """
        保存训练好的模型 (包含随机索引)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存均值和协方差
        np.save(os.path.join(output_dir, 'means.npy'), self.means)
        np.save(os.path.join(output_dir, 'covs.npy'), self.covs)
        
        # 保存配置信息
        config = {
            'image_size': self.image_size,
            'device': self.device,
            'means_shape': self.means.shape if self.means is not None else None,
            'covs_shape': self.covs.shape if self.covs is not None else None
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # ==========================================
        # 修改：保存随机索引而不是投影器
        # ==========================================
        if self.selected_indices is not None:
            # 保存为 Tensor 文件
            torch.save(self.selected_indices.cpu(), os.path.join(output_dir, 'selected_indices.pt'))
            print(f"💾 随机索引已保存: selected_indices.pt")
        
        print(f"✅ 模型已保存到: {output_dir}")
    


    def load_model(self, model_dir):
        """
        加载训练好的模型
        """
        self.means = np.load(os.path.join(model_dir, 'means.npy'))
        self.covs = np.load(os.path.join(model_dir, 'covs.npy'))
        
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.image_size = config['image_size']
        
        # 加载投影器（如果存在）
        projector_path = os.path.join(model_dir, 'projector.pkl')
        if os.path.exists(projector_path):
            import joblib
            self.projector = joblib.load(projector_path)
        
        print(f"✅ 模型从 {model_dir} 加载成功")
        print(f"📊 模型信息:")
        print(f"   - 均值矩阵形状: {self.means.shape}")
        print(f"   - 协方差矩阵形状: {self.covs.shape}")
        print(f"   - 图像尺寸: {self.image_size}")

# 测试代码
if __name__ == "__main__":
    # 简单的功能测试
    print("PaDiM Trainer 模块测试")
    trainer = PaDiMTrainer()
    print(f"设备: {trainer.device}")
    print("模块加载成功!")
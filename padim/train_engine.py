import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from PIL import Image

# ==========================================
# 1. 核心：端到端 ONNX 包装器
#    包含：归一化 -> Backbone -> 拼接 -> 降维
# ==========================================
class PaDiMOnnxWrapper(nn.Module):
    def __init__(self, feature_extractor, indices=None):
        super(PaDiMOnnxWrapper, self).__init__()
        self.backbone = feature_extractor
        
        # 注册通道选择索引 (如果有)
        if indices is not None:
            self.register_buffer('indices', indices)
        else:
            self.indices = None

        # ImageNet 标准均值和方差
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # 1. 归一化 (输入假设是 0~1 的 float)
        x = (x - self.mean) / self.std
        
        # 2. 提取特征 [x1, x2, x3]
        features_list = self.backbone(x)
        
        # 3. 多尺度融合 (以 layer1 为基准尺寸)
        # layer1 shape: [B, 64, H/4, W/4]
        base_feat = features_list[0]
        target_h, target_w = base_feat.shape[2], base_feat.shape[3]
        
        resized_features = [base_feat]
        for f in features_list[1:]:
            # 双线性插值上采样
            f = F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_features.append(f)
        
        # 拼接: [B, C_total, H, W]
        out = torch.cat(resized_features, dim=1)
        
        # 4. 维度转换 [B, C, H, W] -> [B, H, W, C]
        # 方便 C++ 处理
        out = out.permute(0, 2, 3, 1)
        
        # 5. 随机通道选择 (降维)
        if self.indices is not None:
            out = torch.index_select(out, 3, self.indices)
            
        return out

# ==========================================
# 2. 基础特征提取器 (ResNet18)
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(*list(model.children())[:5])
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        return [x1, x2, x3]

# ==========================================
# 3. 训练引擎
# ==========================================
class PaDiMTrainer:
    def __init__(self, data_path, save_path, batch_size=8, image_size=112, reduce_dims=100, device='cuda'):
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.reduce_dims = reduce_dims
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.save_path, exist_ok=True)

    def _get_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        class FlatFolderDataset(torch.utils.data.Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                self.transform = transform
                self.samples = [os.path.join(root, f) for f in os.listdir(root) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
                if not self.samples: raise RuntimeError(f"在 {root} 中未找到任何图片文件")
            def __len__(self): return len(self.samples)
            def __getitem__(self, i):
                with open(self.samples[i], 'rb') as f:
                    img = Image.open(f).convert('RGB')
                return self.transform(img) if self.transform else img, 0
        
        return DataLoader(FlatFolderDataset(self.data_path, transform), batch_size=self.batch_size, shuffle=False)

    def _embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        y = F.interpolate(y, size=(H1, W1), mode='bilinear', align_corners=False)
        return torch.cat([x, y], dim=1)

    def train(self):
        print(f"[Train] 初始化 (Device: {self.device}, Image Size: {self.image_size})...")
        model = FeatureExtractor().to(self.device)
        model.eval()
        dataloader = self._get_dataloader()

        # --- 1. 提取特征 (Python 端计算统计量用) ---
        embedding_vectors = []
        print("[Train] 提取特征...")
        for imgs, _ in tqdm(dataloader):
            imgs = imgs.to(self.device)
            features = model(imgs)
            f_map = features[0]
            for f in features[1:]:
                f_map = self._embedding_concat(f_map, f)
            embedding_vectors.append(f_map.cpu())

        embedding_vectors = torch.cat(embedding_vectors, dim=0)
        N, C_total, H, W = embedding_vectors.size()
        print(f"[Train] 原始特征: {embedding_vectors.shape}")

        # --- 2. 生成随机索引并降维 ---
        selected_indices = None
        if self.reduce_dims < C_total:
            print(f"[Train] 降维: {C_total} -> {self.reduce_dims}")
            # 生成随机索引
            selected_indices = torch.randperm(C_total)[:self.reduce_dims]
            # 对 Python 端的特征进行切片
            embedding_vectors = torch.index_select(embedding_vectors, 1, selected_indices)
            C = self.reduce_dims
        else:
            C = C_total

        # --- 3. 计算统计量 ---
        print("[Train] 计算均值协方差...")
        # 变换形状: (N, C, H, W) -> (N, H, W, C) -> (N, H*W, C)
        embedding_vectors = embedding_vectors.permute(0, 2, 3, 1).reshape(N, H * W, C)
        
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        
        cov_inv_list = []
        embeddings_np = embedding_vectors.numpy()
        identity = 0.01 * np.eye(C)
        for i in tqdm(range(H * W)):
            cov = np.cov(embeddings_np[:, i, :], rowvar=False) + identity
            cov_inv_list.append(np.linalg.inv(cov))
        cov_inv = np.stack(cov_inv_list, axis=0).reshape(H, W, C, C)

        # --- 4. 保存二进制文件 ---
        print(f"[Export] 保存至: {self.save_path}")
        mean.astype(np.float32).tofile(os.path.join(self.save_path, 'means.bin'))
        cov_inv.astype(np.float32).tofile(os.path.join(self.save_path, 'inv_covs.bin'))
        
        # 保存 Config (统一格式)
        with open(os.path.join(self.save_path, 'config.txt'), 'w') as f:
            f.write(f"input_width={self.image_size}\n")
            f.write(f"input_height={self.image_size}\n")
            f.write(f"feature_map_h={H}\n")
            f.write(f"feature_map_w={W}\n")
            f.write(f"feature_dim={C}\n")

        # --- 5. 导出 ONNX (使用 PaDiMOnnxWrapper) ---
        print("[Export] 导出 ONNX (包含归一化+拼接+降维)...")
        
        if selected_indices is not None:
            selected_indices = selected_indices.to(self.device)
            
        # 实例化 Wrapper，传入模型和索引
        wrapper = PaDiMOnnxWrapper(model, selected_indices)
        wrapper.eval().to(self.device)

        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        onnx_path = os.path.join(self.save_path, 'padim_backbone.onnx')
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=['input'],
            output_names=['features'], # 输出直接是最终特征图 [B, H, W, C]
            dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
        )
        
        # 验证输出形状
        with torch.no_grad():
            out_tensor = wrapper(dummy_input)
            print(f"   模型输出形状: {out_tensor.shape} (期望: [1, {H}, {W}, {C}])")
            
        print("[Success] 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--reduce_dims', type=int, default=100)
    args = parser.parse_args()

    PaDiMTrainer(args.data, args.out, args.batch_size, args.image_size, args.reduce_dims).train()
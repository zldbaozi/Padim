import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from torchvision.datasets import DatasetFolder
from PIL import Image
# ==========================================
# 新增：ONNX 包装器 (嵌入归一化)
# ==========================================
class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super(OnnxWrapper, self).__init__()
        self.model = model
        # ImageNet 标准均值和方差
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # 假设输入 x 是 [0, 1] 范围的 float 张量
        # 在模型内部进行归一化
        x = (x - self.mean) / self.std
        return self.model(x)
    

# ==========================================
# 1. 特征提取器定义
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 加载预训练模型
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 截取前三层
        self.layer1 = nn.Sequential(*list(model.children())[:5])
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        return [x1, x2, x3]

# ==========================================
# 2. 训练引擎
# ==========================================
class PaDiMTrainer:
    def __init__(self, data_path, save_path, batch_size=16, image_size=112, device='cuda'):
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(self.save_path, exist_ok=True)

    def _get_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 自定义数据集，直接读取文件夹下的图片，不需要子文件夹
        class FlatFolderDataset(torch.utils.data.Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                self.transform = transform
                self.samples = [os.path.join(root, fname) for fname in os.listdir(root) 
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
                if len(self.samples) == 0:
                    raise RuntimeError(f"在 {root} 中未找到任何图片文件")

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                path = self.samples[index]
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, 0  # 返回图片和一个虚拟标签

        dataset = FlatFolderDataset(self.data_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

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
        print(f"[Train] 数据集: {len(dataloader.dataset)} 张图片")

        # --- 1. 提取特征 ---
        embedding_vectors = []
        print("[Train] 正在提取特征...")
        for imgs, _ in tqdm(dataloader):
            imgs = imgs.to(self.device)
            features = model(imgs)
            
            # 拼接特征
            f_map = features[0]
            for f in features[1:]:
                f_map = self._embedding_concat(f_map, f)
            embedding_vectors.append(f_map.cpu())

        embedding_vectors = torch.cat(embedding_vectors, dim=0) # (N, C, H, W)
        
        # --- 2. 计算统计量 ---
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        print("[Train] 计算均值和协方差矩阵...")
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        
        cov_inv_list = []
        embeddings_np = embedding_vectors.numpy()
        identity = 0.01 * np.eye(C) # 正则项
        
        for i in tqdm(range(H * W)):
            sample = embeddings_np[:, i, :]
            cov = np.cov(sample, rowvar=False) + identity
            cov_inv_list.append(np.linalg.inv(cov))
            
        cov_inv = np.stack(cov_inv_list, axis=0).reshape(H, W, C, C)

        # --- 3. 保存结果 ---
        print(f"[Export] 保存结果至: {self.save_path}")
        
        # 保存二进制文件 (C++用)
        mean.astype(np.float32).tofile(os.path.join(self.save_path, 'means.bin'))
        cov_inv.astype(np.float32).tofile(os.path.join(self.save_path, 'inv_covs.bin'))
        
        # 保存 Python 格式 (备用)
        np.savez(os.path.join(self.save_path, 'params.npz'), mean=mean, cov_inv=cov_inv)

        # 保存 Config
        with open(os.path.join(self.save_path, 'config.txt'), 'w') as f:
            f.write(f"image_size={self.image_size}\n")
            f.write(f"channels={C}\n")
            f.write(f"height={H}\n")
            f.write(f"width={W}\n")

        # 导出 ONNX
        self._export_onnx(model)
        print("[Success] 训练完成！")

    def _export_onnx(self, model):
        print("[Export] 正在封装模型以包含归一化参数...")
        
        # 使用 Wrapper 封装模型
        wrapper = OnnxWrapper(model)
        wrapper.eval()
        wrapper.to(self.device)

        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        onnx_path = os.path.join(self.save_path, 'padim_backbone.onnx')
        
        torch.onnx.export(
            wrapper,  # 导出封装后的模型
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=['input'],
            output_names=['layer1', 'layer2', 'layer3'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        print(f"[Export] ONNX 模型已保存 (内含归一化): {onnx_path}")

# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaDiM 训练工具")
    parser.add_argument('--data', type=str, required=True, help='训练数据路径 (需包含子文件夹)')
    parser.add_argument('--out', type=str, required=True, help='输出路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--image_size', type=int, default=112, help='输入图片大小')
    
    args = parser.parse_args()

    trainer = PaDiMTrainer(
        data_path=args.data,
        save_path=args.out,
        batch_size=args.batch_size,
        image_size=args.image_size
        )
    trainer.train()
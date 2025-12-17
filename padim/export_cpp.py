import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.padim_detector import PaDiMDetector

class PaDiMOnnxWrapper(nn.Module):
    """
    将 归一化 + 特征提取 + 多尺度融合 + 随机降维 封装成一个端到端的模型
    输入: [1, 3, 112, 112] (RGB, 0~1 float 或 0~255 uint8，转为 float)
    输出: [1, 28, 28, 100]
    """
    def __init__(self, feature_extractor, indices):
        super().__init__()
        self.backbone = feature_extractor
        # 将 indices 注册为 buffer，这样它会被保存但不会被视为模型参数
        self.register_buffer('indices', indices)

        # ==========================================
        # 👇👇👇 新增：嵌入归一化参数 👇👇👇
        # ==========================================
        # ImageNet 标准均值和方差 (如果你的训练数据用的是这个)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, x):
        # ==========================================
        # 👇👇👇 新增：模型内部归一化 👇👇👇
        # ==========================================
        # 假设输入 x 是 0~1 的 float (C++ 传进来时只需 /255.0)
        x = (x - self.mean) / self.std
        
        # 1. 提取特征 (List of tensors)
        features_list = self.backbone(x)
        
        # 2. 多尺度融合 (Resize & Concat)
        # 假设 features_list[0] 是最大的 (28x28)
        target_h, target_w = features_list[0].shape[2], features_list[0].shape[3]
        
        resized_features = []
        for f in features_list:
            # 只有尺寸不对时才插值
            if f.shape[2] != target_h or f.shape[3] != target_w:
                f = F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_features.append(f)
        
        # [B, C_total, H, W]
        out = torch.cat(resized_features, dim=1)
        
        # 3. 维度转换 [B, C, H, W] -> [B, H, W, C]
        # 这样做是为了方便 C++ 后续计算 (H*W 个像素并行)
        out = out.permute(0, 2, 3, 1)
        
        # 4. 随机通道选择 (降维)
        # 直接在模型内部完成，C++ 就不需要知道 indices 了
        if self.indices is not None:
            out = torch.index_select(out, 3, self.indices)
            
        return out

def export_to_cpp(model_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"正在加载 Python 模型: {model_dir}")
    detector = PaDiMDetector(model_dir=model_dir)
    device = torch.device('cpu') # 导出时用 CPU 即可
    detector.feature_extractor.to(device)
    
    # ==========================================
    # 1. 导出 ONNX 模型 (ResNet + 预处理)
    # ==========================================
    print("-" * 50)
    print("正在导出 ONNX 模型...")
    
    indices = detector.indices.cpu() if detector.indices is not None else None
    wrapper = PaDiMOnnxWrapper(detector.feature_extractor, indices)
    wrapper.eval()


    dummy_input = torch.randn(1, 3, 112, 112) 
    
    onnx_path = os.path.join(output_dir, "padim_backbone.onnx")
    
   
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['features'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
    )
    print(f"✅ FP32 ONNX 模型已保存: {onnx_path}")
    
    # 验证一下输出形状
    with torch.no_grad():
        out_tensor = wrapper(dummy_input)
        print(f"   模型输出形状: {out_tensor.shape} (期望: [1, 28, 28, 100])")

    # ==========================================
    # 2. 导出统计参数 (均值 & 逆协方差)
    # ==========================================
    print("-" * 50)
    print("正在计算并导出统计参数 (.bin)...")
    
    # 获取均值 [H, W, C]
    means = detector.means.cpu().numpy().astype(np.float32)
    
    # 获取协方差并计算逆矩阵 [H, W, C, C]
    covs = detector.covs.cpu().numpy().astype(np.float32)
    H, W, C, _ = covs.shape
    
    print("   正在计算伪逆矩阵 (这可能需要一点时间)...")
    # 展平计算
    covs_flat = covs.reshape(-1, C, C)
    inv_covs_flat = np.linalg.pinv(covs_flat) # 使用伪逆保证稳定性
    inv_covs = inv_covs_flat.reshape(H, W, C, C).astype(np.float32)
    
    # 保存为二进制文件
    means_path = os.path.join(output_dir, "means.bin")
    inv_covs_path = os.path.join(output_dir, "inv_covs.bin")
    
    means.tofile(means_path)
    inv_covs.tofile(inv_covs_path)
    
    print(f"✅ 均值文件已保存: {means_path} | Size: {means.nbytes / 1024:.2f} KB")
    print(f"✅ 逆协方差已保存: {inv_covs_path} | Size: {inv_covs.nbytes / 1024 / 1024:.2f} MB")
    
    # ==========================================
    # 3. 生成 C++ 配置文件
    # ==========================================
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"input_width=112\n")
        f.write(f"input_height=112\n")
        f.write(f"feature_map_h={H}\n")
        f.write(f"feature_map_w={W}\n")
        f.write(f"feature_dim={C}\n")
    print(f"✅ 配置文件已生成: {config_path}")
    
    print("-" * 50)
    print("导出完成！请将 cpp_model 文件夹复制到你的 C++ 项目中。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Python模型路径')
    parser.add_argument('--output_dir', type=str, default='./cpp_model', help='导出路径')
    args = parser.parse_args()
    
    export_to_cpp(args.model_dir, args.output_dir)
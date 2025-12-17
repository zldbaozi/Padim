import torch
import time
import numpy as np
from torchvision.models import resnet18, wide_resnet50_2

def benchmark_fov_feature_extraction(
    arch='resnet18', 
    img_h=2048, 
    img_w=2448, 
    device='cuda', 
    loops=50
):
    """
    基准测试：测量大图输入到 CNN 提取特征的纯耗时
    """
    print(f"\n🚀 [基准测试启动] 架构: {arch} | 输入分辨率: {img_w}x{img_h}")
    
    # 1. 加载模型 (不加载预训练权重，只测计算速度)
    try:
        if arch == 'resnet18':
            model = resnet18(pretrained=False)
        elif arch == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=False)
        else:
            print("❌ 不支持的模型架构")
            return
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 注册 Hook (模拟 PaDiM 提取中间层特征的行为)
    # PaDiM 必须提取 layer1, layer2, layer3，这会增加显存读写开销
    features = []
    def hook(module, input, output):
        features.append(output) # 只是引用，不拷贝

    model.layer1.register_forward_hook(hook)
    model.layer2.register_forward_hook(hook)
    model.layer3.register_forward_hook(hook)

    if torch.cuda.is_available() and device == 'cuda':
        model.to(device)
        model.eval()
    else:
        print("⚠️ CUDA 不可用，切换到 CPU 模式 (速度会很慢)")
        device = 'cpu'
        model.to('cpu')
        model.eval()

    # 3. 创建虚拟大图 (Batch Size = 1)
    try:
        # 模拟一张 RGB 图片 (B, C, H, W)
        dummy_input = torch.randn(1, 3, img_h, img_w).to(device)
        
        # 计算显存占用
        mem_mb = dummy_input.element_size() * dummy_input.nelement() / (1024 * 1024)
        print(f"✅ 输入张量创建成功 | 显存占用: {mem_mb:.2f} MB")
        
    except RuntimeError as e:
        print(f"❌ 显存不足，无法创建输入张量: {e}")
        print("建议: 减小 img_h 和 img_w 的值")
        return

    # 4. 预热 (Warm-up)
    print("🔥 正在预热 GPU (消除初始化抖动)...")
    try:
        with torch.no_grad():
            for _ in range(5):
                features = []
                _ = model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"❌ 预热阶段爆显存了: {e}")
        return

    # 5. 正式测试
    print(f"⏱️  开始测试 (循环 {loops} 次)...")
    timings = []
    
    try:
        with torch.no_grad():
            for i in range(loops):
                features = [] # 清空列表
                
                if device == 'cuda':
                    torch.cuda.synchronize() # 同步起点
                
                start = time.perf_counter()
                
                # === 核心过程 ===
                _ = model(dummy_input)
                # =============
                
                if device == 'cuda':
                    torch.cuda.synchronize() # 同步终点
                
                end = time.perf_counter()
                
                timings.append((end - start) * 1000) # ms
                print(f"\r进度: {i+1}/{loops}", end="")
                
    except RuntimeError as e:
        print(f"\n❌ 测试过程中爆显存: {e}")
        return

    # 6. 结果统计
    avg_time = np.mean(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    fps = 1000 / avg_time
    
    print(f"\n\n{'='*40}")
    print(f"📊 测试结果报告")
    print(f"{'='*40}")
    print(f"   模型架构 : {arch}")
    print(f"   图像尺寸 : {img_w} x {img_h}")
    print(f"   平均耗时 : {avg_time:.2f} ms")
    print(f"   最快耗时 : {min_time:.2f} ms")
    print(f"   最慢耗时 : {max_time:.2f} ms")
    print(f"   FPS      : {fps:.2f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    # ==========================================
    # 在这里修改你的实际大图分辨率
    # ==========================================
    
    # 场景 1: 500万像素工业相机 (2448 x 2048)
    benchmark_fov_feature_extraction(
        arch='resnet18', 
        img_h=281, 
        img_w=225
    )
    
    # 场景 2: 4K 分辨率 (3840 x 2160) - 如果显存够大可以取消注释
    # benchmark_fov_feature_extraction(
    #     arch='resnet18', 
    #     img_h=2160, 
    #     img_w=3840
    # )
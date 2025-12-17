import os
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 1. 基础工具 (解决中文路径问题)
# ==========================================

def cv_imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        return None

def cv_imwrite(path, img):
    try:
        ext = os.path.splitext(path)[1] or '.png'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except:
        pass
    return False

# 定义一个只做 ToTensor 和 Normalize 的 transform，保留原尺寸
def get_raw_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
# ==========================================
# 2. 推理引擎 (整图推理)
# ==========================================

class PaDiMInferenceEngine:
    def __init__(self, model_dir, device='cuda'):
        from models.padim_detector import PaDiMDetector
        
        print(f"🏗️  加载模型: {model_dir}")
        self.detector = PaDiMDetector(model_dir=model_dir)
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_folder(self, input_dir, save_dir, threshold=3.0):
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"❌ 输入目录不存在: {input_dir}")
            return
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        if not files:
            print("⚠️ 文件夹为空")
            return
        print(f"🚀 开始处理 {len(files)} 张图片...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 预热 GPU
        print("🔥 正在预热 GPU...")
        dummy = torch.randn(1, 3, 112, 112).to(self.device)  # 确保尺寸为 112×112
        for _ in range(5): self.detector.predict(dummy)
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        
        for idx, f in enumerate(files):
            img_path = os.path.join(input_dir, f)
            img_bgr = cv_imread(img_path)
            if img_bgr is None: 
                print(f"⚠️ 无法读取图片: {img_path}")
                continue
            
            # 1. 预处理
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # 2. 推理
            amap, score = self.detector.predict(input_tensor)
            score = score[0]
            amap = amap[0]  # (H, W) 原始异常值
            
            # 3. 打印状态
            status = "🔴异常" if score > threshold else "🟢正常"
            print(f"[{idx+1}/{len(files)}] {f} -> {status} (得分: {score:.2f})")
            
            # ==========================================
            # 4. 灰度热力图生成
            # ==========================================
            
            # Resize 到原图大小
            amap_resized = cv2.resize(amap, (img_bgr.shape[1], img_bgr.shape[0]))
            
            # 归一化策略
            max_val = 4.0  # 固定阈值归一化
            norm_map = np.clip(amap_resized / max_val, 0, 1)
            
            # 转为 8位 灰度图 (0=黑, 255=白)
            heatmap_gray = (norm_map * 255).astype(np.uint8)
            
            # 保存结果
            save_path = os.path.join(save_dir, f)
            cv_imwrite(save_path, heatmap_gray)

        t_end = time.perf_counter()
        avg_time = (t_end - t_start) / len(files) * 1000
        print(f"✅ 全部完成 | 平均耗时: {avg_time:.2f} ms/张")


# ==========================================
# 3. 主入口
# ==========================================

def main(args):
    try:
        # 初始化引擎
        engine = PaDiMInferenceEngine(args.model_dir)
        
        # 执行推理
        engine.process_folder(
            input_dir=args.test_data, 
            save_dir=args.save_dir, 
            threshold=args.threshold
        )
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaDiM Inference (Full Image)")
    
    parser.add_argument('--model_dir', type=str, required=True, help='模型路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据文件夹路径')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存路径')
    parser.add_argument('--threshold', type=float, default=3.0, help='异常阈值')
    
    args = parser.parse_args()
    main(args)
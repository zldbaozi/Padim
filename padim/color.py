import cv2
import numpy as np
import os
import glob

def calculate_bright_ratio(image_path, brightness_threshold=160):
    """
    计算灰度图中亮度超过 brightness_threshold 的像素占比
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    
    # 1. 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 二值化：找出高亮区域 (焊点/反光)
    # 这里的 160 是一个经验值，你可以根据实际图片亮度调整
    # 如果焊点非常亮，可以设高一点 (比如 200)
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # 3. 计算占比
    total_pixels = gray.shape[0] * gray.shape[1]
    bright_pixels = cv2.countNonZero(bright_mask)
    ratio = bright_pixels / total_pixels
    
    return ratio

def main():
    # --- 配置 ---
    sample_dir = "./dataset/industrial/train/cut" 
    max_samples = 50 
    
    # 关键参数：定义什么是"白色/高亮"
    # 0-255 之间，越接近 255 越亮
    # 建议尝试 150, 180, 200
    BRIGHTNESS_THRESH = 160 
    # -----------

    print(f"正在分析 {sample_dir} 中的样本 (亮度阈值 > {BRIGHTNESS_THRESH})...")
    
    image_files = glob.glob(os.path.join(sample_dir, "*.png")) + \
                  glob.glob(os.path.join(sample_dir, "*.bmp")) + \
                  glob.glob(os.path.join(sample_dir, "*.jpg"))
    
    if not image_files:
        print("❌ 未找到图片")
        return

    ratios = []
    print("-" * 40)
    print(f"{'文件名':<30} | {'高亮占比':<10}")
    print("-" * 40)

    for i, img_path in enumerate(image_files):
        if i >= max_samples: break
        
        ratio = calculate_bright_ratio(img_path, BRIGHTNESS_THRESH)
        ratios.append(ratio)
        
        filename = os.path.basename(img_path)
        print(f"{filename:<30} | {ratio:.4f}")

    if not ratios:
        return

    avg_ratio = np.mean(ratios)
    min_ratio = np.min(ratios)
    max_ratio = np.max(ratios)
    std_dev = np.std(ratios)

    print("-" * 40)
    print("📊 统计结果:")
    print(f"  亮度阈值: > {BRIGHTNESS_THRESH}")
    print(f"  平均占比: {avg_ratio:.4f}")
    print(f"  最小占比: {min_ratio:.4f}")
    print(f"  最大占比: {max_ratio:.4f}")
    print("-" * 40)
    
    thresh_conservative = min_ratio * 0.8
    
    print("💡 建议 cut.py 设置:")
    print(f"  1. 灰度阈值 (threshold): {BRIGHTNESS_THRESH}")
    print(f"  2. 占比阈值 (ratio): > {thresh_conservative:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
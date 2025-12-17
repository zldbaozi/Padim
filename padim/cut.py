import os
import cv2
import numpy as np

def filter_by_color(patch_img):
    """
    基于 HSV 的银色/白色过滤器
    逻辑：低饱和度(S) + 中高亮度(V) = 银色/白色/灰色
    """
    # 1. 转 HSV
    hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
    
    # 2. 定义银色/白色的范围
    # H: 0-180 (任意色调)
    # S: 0-60 (关键！放宽到60，只要不是鲜艳的黄色/绿色都算)
    # V: 50-255 (排除黑色背景)
    lower_silver = np.array([0, 0, 50])
    upper_silver = np.array([180, 60, 255])
    
    # 3. 生成掩码
    mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # 4. 计算占比
    total_pixels = patch_img.shape[0] * patch_img.shape[1]
    silver_pixels = cv2.countNonZero(mask)
    ratio = silver_pixels / total_pixels
    
    # 5. 设定阈值
    # 根据你之前的统计，好的图片银色占比约为 0.02 - 0.04
    # 这里设为 0.015 (1.5%) 作为底线，保证不漏掉好的
    return ratio > 0.010

def extract_aligned_patches(img, patch_size=(225, 281), debug=False):
    """
    基于【自适应阈值 + 定向边界剔除 + 补边 + HSV银色特征过滤】的自动对齐切割
    """
    h_img, w_img = img.shape[:2]
    target_w, target_h = patch_size
    
    # 1. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自适应阈值 + 反转 
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        19, 
        10
    )

    # 3. 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    min_area = 1500 
    max_area = (h_img * w_img) / 5 

    if debug:
        debug_img = img.copy()
        h, w = binary.shape[:2]
        scale = 1000/max(h,w)
        cv2.imshow("Binary Debug", cv2.resize(binary, (0,0), fx=scale, fy=scale))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- 定向边界剔除 ---
            # 剔除贴着 左、上、下 边缘的残缺品
            # 保留贴着 右 边缘的（后续补边）
            if x < 5 or y < 5 or (y + h) > (h_img - 5):
                if debug:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                continue

            aspect_ratio = float(w) / h
            
            if 0.25 < aspect_ratio < 4.0: 
                
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                x1 = cx - target_w // 2
                y1 = cy - target_h // 2
                x2 = x1 + target_w
                y2 = y1 + target_h
                
                # --- 自动补边 (Padding) ---
                patch_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                src_x1 = max(0, x1); src_y1 = max(0, y1)
                src_x2 = min(w_img, x2); src_y2 = min(h_img, y2)
                
                dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                valid_h = src_y2 - src_y1
                valid_w = src_x2 - src_x1
                
                if valid_h > 10 and valid_w > 10:
                    patch_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
                    
                    # --- HSV 颜色过滤 ---
                    if filter_by_color(patch_canvas):
                        patches.append(patch_canvas)
                        if debug:
                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                    else:
                        if debug:
                            # 银色占比不够（可能是纯黑背景或纯黄背景）
                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if debug:
        h, w = debug_img.shape[:2]
        scale = 1000/max(h,w)
        cv2.imshow("Result Debug (Green=Keep, Purple=Edge Skip)", cv2.resize(debug_img, (0,0), fx=scale, fy=scale))
        print("按任意键继续...")
        cv2.waitKey(0)

    return patches

def process_dataset(src_dir, dst_dir, patch_size=(225, 281)):
    os.makedirs(dst_dir, exist_ok=True)
    exts = (".bmp", ".png", ".jpg", ".jpeg")
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    if not files:
        print("源目录无图像")
        return

    total_patches = 0
    DEBUG_MODE = False 
    
    for f in files:
        path = os.path.join(src_dir, f)
        img = cv2.imread(path)
        if img is None: continue
            
        print(f"正在处理: {f} ...")
        patches = extract_aligned_patches(img, patch_size=patch_size, debug=DEBUG_MODE)
        
        base = os.path.splitext(f)[0]
        for i, patch in enumerate(patches):
            out_name = f"{base}_{i:03d}.png"
            cv2.imwrite(os.path.join(dst_dir, out_name), patch)
        
        print(f"  -> 提取了 {len(patches)} 个补丁")
        total_patches += len(patches)
        
    if DEBUG_MODE:
        cv2.destroyAllWindows()
        
    print(f"全部完成。共生成 {total_patches} 个训练样本。")

if __name__ == "__main__":
    source_dir = "./dataset/industrial/train"      
    target_dir = "./dataset/industrial/train/cut"
    
    # 你的# filepath: c:\Users\mento\Desktop\padim_project\cut.py
import os
import cv2
import numpy as np

def filter_by_color(patch_img):
    """
    基于 HSV 的银色/白色过滤器
    逻辑：低饱和度(S) + 中高亮度(V) = 银色/白色/灰色
    """
    # 1. 转 HSV
    hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
    
    # 2. 定义银色/白色的范围
    # H: 0-180 (任意色调)
    # S: 0-60 (关键！放宽到60，只要不是鲜艳的黄色/绿色都算)
    # V: 50-255 (排除黑色背景)
    lower_silver = np.array([0, 0, 50])
    upper_silver = np.array([180, 60, 255])
    
    # 3. 生成掩码
    mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # 4. 计算占比
    total_pixels = patch_img.shape[0] * patch_img.shape[1]
    silver_pixels = cv2.countNonZero(mask)
    ratio = silver_pixels / total_pixels
    
    # 5. 设定阈值
    # 根据你之前的统计，好的图片银色占比约为 0.02 - 0.04
    # 这里设为 0.015 (1.5%) 作为底线，保证不漏掉好的
    return ratio > 0.015

def extract_aligned_patches(img, patch_size=(225, 281), debug=False):
    """
    基于【自适应阈值 + 定向边界剔除 + 补边 + HSV银色特征过滤】的自动对齐切割
    """
    h_img, w_img = img.shape[:2]
    target_w, target_h = patch_size
    
    # 1. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自适应阈值 + 反转 
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        19, 
        10
    )

    # 3. 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    min_area = 1500 
    max_area = (h_img * w_img) / 5 

    if debug:
        debug_img = img.copy()
        h, w = binary.shape[:2]
        scale = 1000/max(h,w)
        cv2.imshow("Binary Debug", cv2.resize(binary, (0,0), fx=scale, fy=scale))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- 定向边界剔除 ---
            # 剔除贴着 左、上、下 边缘的残缺品
            # 保留贴着 右 边缘的（后续补边）
            if x < 5 or y < 5 or (y + h) > (h_img - 5):
                if debug:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                continue

            aspect_ratio = float(w) / h
            
            if 0.25 < aspect_ratio < 4.0: 
                
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                x1 = cx - target_w // 2
                y1 = cy - target_h // 2
                x2 = x1 + target_w
                y2 = y1 + target_h
                
                # --- 自动补边 (Padding) ---
                patch_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                src_x1 = max(0, x1); src_y1 = max(0, y1)
                src_x2 = min(w_img, x2); src_y2 = min(h_img, y2)
                
                dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                valid_h = src_y2 - src_y1
                valid_w = src_x2 - src_x1
                
                if valid_h > 10 and valid_w > 10:
                    patch_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
                    
                    # --- HSV 颜色过滤 ---
                    if filter_by_color(patch_canvas):
                        patches.append(patch_canvas)
                        if debug:
                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                    else:
                        if debug:
                            # 银色占比不够（可能是纯黑背景或纯黄背景）
                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if debug:
        h, w = debug_img.shape[:2]
        scale = 1000/max(h,w)
        cv2.imshow("Result Debug (Green=Keep, Purple=Edge Skip)", cv2.resize(debug_img, (0,0), fx=scale, fy=scale))
        print("按任意键继续...")
        cv2.waitKey(0)

    return patches

def process_dataset(src_dir, dst_dir, patch_size=(225, 281)):
    os.makedirs(dst_dir, exist_ok=True)
    exts = (".bmp", ".png", ".jpg", ".jpeg")
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    if not files:
        print("源目录无图像")
        return

    total_patches = 0
    DEBUG_MODE = False 
    
    for f in files:
        path = os.path.join(src_dir, f)
        img = cv2.imread(path)
        if img is None: continue
            
        print(f"正在处理: {f} ...")
        patches = extract_aligned_patches(img, patch_size=patch_size, debug=DEBUG_MODE)
        
        base = os.path.splitext(f)[0]
        for i, patch in enumerate(patches):
            out_name = f"{base}_{i:03d}.png"
            cv2.imwrite(os.path.join(dst_dir, out_name), patch)
        
        print(f"  -> 提取了 {len(patches)} 个补丁")
        total_patches += len(patches)
        
    if DEBUG_MODE:
        cv2.destroyAllWindows()
        
    print(f"全部完成。共生成 {total_patches} 个训练样本。")

if __name__ == "__main__":
    source_dir = "./dataset/industrial/train"      
    target_dir = "./dataset/industrial/train/cut"
    
    # 你的尺寸
    PATCH_SIZE = (225, 281) 
    
    process_dataset(source_dir, target_dir, patch_size=PATCH_SIZE)
import os
import cv2
import numpy as np
import argparse

def cv_imread(path):
    """支持中文路径读取"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取失败: {path}")
        return None

def cv_imwrite(path, img):
    """支持中文路径保存"""
    try:
        ext = os.path.splitext(path)[1] or '.png'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception as e:
        print(f"保存失败: {path}")
    return False

class ComponentCropper:
    def __init__(self, template_path, match_threshold=0.7):
        self.template = cv_imread(template_path)
        if self.template is None:
            raise ValueError(f"无法读取模板: {template_path}")
        self.h, self.w = self.template.shape[:2]
        self.threshold = match_threshold
        print(f"✅ 模板加载成功: {self.w}x{self.h}")

    def crop_from_folder(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📂 创建输出目录: {output_dir}")

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        total_crops = 0
        
        for idx, filename in enumerate(files):
            full_path = os.path.join(input_dir, filename)
            large_img = cv_imread(full_path)
            
            if large_img is None: continue
            
            print(f"[{idx+1}/{len(files)}] 正在处理: {filename} ...", end="")
            
            # 1. 模板匹配
            res = cv2.matchTemplate(large_img, self.template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.threshold)
            
            # 2. 坐标去重 (NMS)
            rects = []
            for pt in zip(*loc[::-1]):
                rects.append([int(pt[0]), int(pt[1]), self.w, self.h])
            rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)
            
            print(f" -> 发现 {len(rects)} 个目标")
            
            # 3. 裁剪并保存 (带坐标)
            base_name = os.path.splitext(filename)[0]
            for (x, y, w, h) in rects:
                # 边界检查
                if y < 0 or x < 0 or y+h > large_img.shape[0] or x+w > large_img.shape[1]:
                    continue
                
                crop = large_img[y:y+h, x:x+w]
                
                # ==========================================
                # 核心修改：文件名包含坐标信息
                # 格式: 原名__x_{x}_y_{y}.png
                # ==========================================
                save_name = f"{base_name}__x_{x}_y_{y}.png"
                save_path = os.path.join(output_dir, save_name)
                
                cv_imwrite(save_path, crop)
                total_crops += 1

        print(f"\n🎉 全部完成！共裁剪出 {total_crops} 张小图。")
        print(f"📂 保存位置: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量裁剪元器件 (文件名带坐标)")
    parser.add_argument('--input_dir', type=str, required=True, help='包含大图的文件夹')
    parser.add_argument('--output_dir', type=str, required=True, help='保存小图的文件夹')
    parser.add_argument('--template', type=str, required=True, help='模板图片路径')
    parser.add_argument('--threshold', type=float, default=0.7, help='匹配阈值 (0.6-0.9)')
    
    args = parser.parse_args()
    
    cropper = ComponentCropper(args.template, args.threshold)
    cropper.crop_from_folder(args.input_dir, args.output_dir)
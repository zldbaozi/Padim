import os
import cv2
import numpy as np
import random

defect_types = ["引脚变形","裂纹","焊点脱落","污渍"]

# ---------- 工具 ----------
def clip_img(img):
    return np.clip(img,0,255).astype(np.uint8)

def rand_color(bright=True):
    c = np.array([random.randint(120,255) if bright else random.randint(0,120) for _ in range(3)])
    return c.tolist()

# ---------- 缺陷生成 (适配小图版) ----------
def defect_pin_bend(img):
    h,w = img.shape[:2]
    img2 = img.copy()
    # 适配小图：减小形变区域宽度 (w//40 -> w//20)
    bw = random.randint(max(3, w//40), max(5, w//18))
    x0 = random.randint(0, max(1, w-bw-1))
    y0 = 0
    roi = img[y0:h, x0:x0+bw].copy()
    
    # 构造形变
    shift = random.randint(-bw//2, bw//2)
    map_x, map_y = np.meshgrid(np.arange(bw), np.arange(h))
    shear_strength = shift / h
    map_x = map_x + (map_y * shear_strength).astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    if bw > 0 and h > 0:
        warped = cv2.remap(roi, map_x.astype(np.float32), map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img2[y0:h, x0:x0+bw] = warped
        
        # 边缘高亮
        edge = cv2.Canny(warped,50,150)
        edge_col = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        img2[y0:h, x0:x0+bw] = clip_img(img2[y0:h, x0:x0+bw] + edge_col//3)
        
    return img2

def defect_crack(img):
    h,w = img.shape[:2]
    img2 = img.copy()
    # 适配小图：裂纹长度
    length = random.randint(w//5, w//2)
    
    x = random.randint(0,w-1); y = random.randint(0,h-1)
    pts = [(x,y)]
    
    # 减少节点数量，防止在小图上画满
    for _ in range(random.randint(5, 10)):
        angle = random.uniform(0,2*np.pi)
        
        # --- 修复报错的关键逻辑 ---
        # 确保上限至少为 6，防止 randint(5, 4) 这种错误
        max_step = max(6, length // 10)
        step = random.randint(3, max_step) # 下限也稍微调小一点到 3
        # -----------------------
        
        x += int(np.cos(angle)*step)
        y += int(np.sin(angle)*step)
        x = max(0,min(w-1,x)); y = max(0,min(h-1,y))
        pts.append((x,y))
        
    # 画主裂纹 (线条变细适配小图)
    for i in range(len(pts)-1):
        cv2.line(img2, pts[i], pts[i+1], (30,30,30), random.randint(1,2), cv2.LINE_AA)
        
    # 分支
    for p in pts[::2]:
        if random.random() > 0.5: continue
        ang = random.uniform(0,2*np.pi)
        l = random.randint(5, 15) # 分支长度减小
        x2 = max(0,min(w-1, p[0]+int(np.cos(ang)*l)))
        y2 = max(0,min(h-1, p[1]+int(np.sin(ang)*l)))
        cv2.line(img2, p, (x2,y2), (40,40,40), 1, cv2.LINE_AA)
        
    return img2

def defect_solder_missing(img):
    h,w = img.shape[:2]
    img2 = img.copy()
    # 适配小图：减小缺失区域
    rw = random.randint(max(5, w//20), max(10, w//8))
    rh = random.randint(max(5, h//20), max(10, h//8))
    
    x0 = random.randint(0, max(1, w-rw-1))
    y0 = random.randint(0, max(1, h-rh-1))
    
    region = img2[y0:y0+rh, x0:x0+rw].copy()
    
    # 去色 + 降亮度
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = (gray * random.uniform(0.4,0.7)).astype(np.uint8)
    
    # 添加噪声
    noise = np.random.randint(0,50,(rh,rw)).astype(np.uint8)
    gray = clip_img(gray + noise)
    region_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 边缘加深
    border = cv2.Canny(region_out,40,120)
    region_out[border>0] = (0,0,0)
    img2[y0:y0+rh, x0:x0+rw] = region_out
    return img2

def perlin_noise(h,w,octaves=4):
    base = np.zeros((h,w),dtype=np.float32)
    for o in range(octaves):
        freq = 2**o
        # 防止除零错误
        sx = max(1,w//(freq*4)); sy = max(1,h//(freq*4))
        small = np.random.rand(sy,sx).astype(np.float32)
        small = cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC)
        base += small / (2**o)
    base = base - base.min()
    base /= (base.max()+1e-6)
    return base

def defect_stain(img):
    h,w = img.shape[:2]
    img2 = img.copy()
    mask = perlin_noise(h,w,octaves=4) # 减少倍频，图案更简单
    
    thresh = np.clip(np.mean(mask)+0.15,0,1)
    stain = (mask>thresh).astype(np.float32)
    
    # 适配小图：减小模糊核 (11->5)
    stain = cv2.GaussianBlur(stain,(5,5),2)
    
    alpha = (stain * random.uniform(0.25,0.55)).reshape(h,w,1)
    color = np.array(rand_color(bright=False),dtype=np.float32)
    overlay = np.ones_like(img2,dtype=np.float32)*color
    blended = img2.astype(np.float32)*(1-alpha)+overlay*alpha
    return clip_img(blended)

# ---------- 主处理 ----------
def apply_defect(img, defect):
    if defect=="引脚变形":
        return defect_pin_bend(img)
    if defect=="裂纹":
        return defect_crack(img)
    if defect=="焊点脱落":
        return defect_solder_missing(img)
    if defect=="污渍":
        return defect_stain(img)
    return img

def process_one(img_path, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        return
    base = os.path.splitext(os.path.basename(img_path))[0]
    for d in defect_types:
        out = apply_defect(img, d)
        out_path = os.path.join(out_dir, f"{base}_{d}.png")
        cv2.imwrite(out_path, out)
        print(f"生成: {out_path}")

if __name__ == "__main__":
    src_dir = "dataset/industrial/train/cut"
    out_dir = "dataset/industrial/test/cut"
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".bmp",".png",".jpg",".jpeg"))]
    
    if not files:
        print(f"错误：源目录 {src_dir} 中没有图片！请先运行 cut.py。")
    else:
        print(f"找到 {len(files)} 张图片，开始生成缺陷样本...")
        for f in files:
            process_one(os.path.join(src_dir,f), out_dir)
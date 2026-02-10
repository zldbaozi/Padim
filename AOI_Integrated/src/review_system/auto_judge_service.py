import time
import os
import shutil
import onnxruntime as ort
import numpy as np
from PIL import Image

# ================= 配置区 =================
# 1. 监听区 (C++ 输出的临时缓冲区)
WATCH_DIR = r"e:\Code\AOI_Integrated\run_data\03_Pending_Review\Pending" 

# 2. 结果归档区 (所有结果都必须去这里)
REAL_NG_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\Real_NG"       # 🔴 确定是坏的
FALSE_ALARM_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\False_Alarm" # 🟢 确定是好的
MANUAL_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\Manual_Review"   # 🟡 机器拿不准，请人工看 (避免 Pending 堆积)

# 3. 模型路径
MODEL_PATH = r"e:\Code\AOI_Integrated\models\resnet\resnet18_review.onnx"
# ==========================================

class SmartJudge:
    def __init__(self, model_path):
        print(f"🚀 加载模型: {model_path} ...")
        # 优先使用 CUDA，如果没有则回退到 CPU
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print("✅ 模型就绪，开始分拣任务...")

    def preprocess(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 1. 补黑边成正方形 (SquarePad) - 必须与训练一致
            w, h = img.size
            max_wh = max(w, h)
            img_padded = Image.new('RGB', (max_wh, max_wh), (0, 0, 0))
            img_padded.paste(img, ((max_wh - w) // 2, (max_wh - h) // 2))
            
            # 2. 调整尺寸至 448x448 - 必须与模型一致
            img = img_padded.resize((448, 448))
            
            img_data = np.array(img).astype('float32')
            img_data /= 255.0
            img_data -= np.array([0.485, 0.456, 0.406]).astype('float32')
            img_data /= np.array([0.229, 0.224, 0.225]).astype('float32')
            img_data = np.transpose(img_data, (2, 0, 1))
            return np.expand_dims(img_data, 0)
        except Exception:
            return None

    def predict(self, img_path):
        data = self.preprocess(img_path)
        # 如果读图失败，返回 0.5/0.5 强制让它进人工复判
        if data is None: return 0.5, 0.5 
        
        outputs = self.session.run(None, {self.input_name: data})
        e_x = np.exp(outputs[0] - np.max(outputs[0]))
        probs = (e_x / e_x.sum(axis=1, keepdims=True))[0]
        return probs[0], probs[1] # [P_False, P_Real]

def run_service():
    # 确保所有目标文件夹都存在
    for d in [WATCH_DIR, REAL_NG_DIR, FALSE_ALARM_DIR, MANUAL_DIR]:
        os.makedirs(d, exist_ok=True)
    
    judge = SmartJudge(MODEL_PATH)
    print(f"👀 正在监听: {WATCH_DIR}")
    print(f"📂 结果将分流至: {os.path.dirname(REAL_NG_DIR)}")

    while True:
        files = [f for f in os.listdir(WATCH_DIR) if f.lower().endswith('.jpg')]
        
        if not files:
            time.sleep(0.5) # 没有文件时休息
            continue
            
        for fname in files:
            fpath = os.path.join(WATCH_DIR, fname)
            
            # 防止文件正在写入时被读取
            try:
                with open(fpath, 'rb'): pass
            except IOError:
                continue 

            p_false, p_real = judge.predict(fpath)
            
            # === 分流策略 ===
            
            # 1. 确信是真缺陷 -> 🔴 Real_NG
            if p_real > 0.85:
                print(f"🔴 真缺陷 ({p_real:.2f}): {fname}")
                try:
                    shutil.move(fpath, os.path.join(REAL_NG_DIR, fname))
                except Exception as e:
                    print(f"移动失败: {e}")

            # 2. 确信是假报警 -> 🟢 False_Alarm
            elif p_false > 0.85:
                print(f"🟢 假报警 ({p_false:.2f}): {fname}")
                try:
                    shutil.move(fpath, os.path.join(FALSE_ALARM_DIR, fname))
                except Exception as e:
                    print(f"移动失败: {e}")
                
            # 3. 模棱两可 -> 🟡 Manual_Review (移出 Pending，避免死循环)
            else:
                print(f"🟡 待定 ({p_real:.2f}): {fname} -> 转人工文件夹")
                try:
                    shutil.move(fpath, os.path.join(MANUAL_DIR, fname))
                except Exception as e:
                    print(f"移动失败: {e}")

        time.sleep(0.1)

if __name__ == "__main__":
    run_service()

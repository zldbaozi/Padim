import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import shutil

# ================= 配置区 =================
# 路径应与 auto_judge_service.py 保持一致
MANUAL_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\Manual_Review"
REAL_NG_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\Real_NG"
FALSE_ALARM_DIR = r"e:\Code\AOI_Integrated\run_data\04_Output_Result\False_Alarm"
# ==========================================

class ManualReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人工复判工具 (Manual Review Helper)")
        self.root.geometry("900x700")

        self.image_list = []
        self.current_image_path = None
        self.tk_image = None # Keep reference to prevent GC
        
        # 确保目标文件夹存在
        for d in [MANUAL_DIR, REAL_NG_DIR, FALSE_ALARM_DIR]:
            os.makedirs(d, exist_ok=True)

        self.setup_ui()
        self.load_next_image()

    def setup_ui(self):
        # 顶部信息栏
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        self.status_label = tk.Label(self.info_frame, text="初始化中...", font=("Microsoft YaHei", 12))
        self.status_label.pack(side="left")
        
        self.count_label = tk.Label(self.info_frame, text="剩余: 0", font=("Microsoft YaHei", 12, "bold"))
        self.count_label.pack(side="right")

        # 图片显示区域
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.image_frame.pack(expand=True, fill="both", padx=10, pady=5)
        
        self.image_label = tk.Label(self.image_frame, text="等待加载图片...", bg="#f0f0f0")
        self.image_label.pack(expand=True)

        # 底部按钮区域
        self.btn_frame = tk.Frame(self.root, height=100)
        self.btn_frame.pack(side="bottom", fill="x", padx=20, pady=20)

        # 按钮: 假报警 (认为是好的) - 对应键盘 Left
        self.btn_false = tk.Button(self.btn_frame, text="✅ 假报警 (False NG)\n[按 ← 键]", 
                                   bg="#90EE90", font=("Microsoft YaHei", 14),
                                   command=lambda: self.move_image('false'))
        self.btn_false.pack(side="left", expand=True, fill="both", padx=10)

        # 按钮: 真缺陷 (认为是坏的) - 对应键盘 Right
        self.btn_true = tk.Button(self.btn_frame, text="❌ 真缺陷 (True NG)\n[按 → 键]", 
                                  bg="#FFB6C1", font=("Microsoft YaHei", 14),
                                  command=lambda: self.move_image('true'))
        self.btn_true.pack(side="right", expand=True, fill="both", padx=10)

        # 键盘快捷键绑定
        self.root.bind("<Left>", lambda e: self.move_image('false'))
        self.root.bind("<Right>", lambda e: self.move_image('true'))

    def load_next_image(self):
        # 刷新列表，只看 jpg, png 等
        all_files = os.listdir(MANUAL_DIR)
        self.image_list = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.image_list.sort() # 按文件名排序

        self.count_label.config(text=f"剩余待判: {len(self.image_list)}")

        if not self.image_list:
            self.display_no_images()
            return

         
        # 获取第一张图片
        filename = self.image_list[0]
        self.current_image_path = os.path.join(MANUAL_DIR, filename)

        try:
            pil_image = Image.open(self.current_image_path)
            
            # 计算缩放比例，适应窗口
            display_w = 800
            display_h = 500
            pil_image.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)
            
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.tk_image, text="")
            self.status_label.config(text=f"当前正在复判: {filename}")
            
            # 启用按钮
            self.btn_false.config(state="normal")
            self.btn_true.config(state="normal")

        except Exception as e:
            print(f"读取图片出错: {e}")
            self.status_label.config(text=f"读取出错: {filename}")
            # 可以选择自动跳过坏文件，这里简单跳过
            self.move_image('skip_error')

    def display_no_images(self):
        self.image_label.config(image="", text="🎉 全部复判完成！\n文件夹为空。", font=("Microsoft YaHei", 20))
        self.status_label.config(text="就绪")
        self.current_image_path = None
        self.btn_false.config(state="disabled")
        self.btn_true.config(state="disabled")

    def move_image(self, decision):
        if not self.current_image_path:
            return

        filename = os.path.basename(self.current_image_path)
        
        # 确定目标目录
        target_dir = None
        if decision == 'false':
            target_dir = FALSE_ALARM_DIR
        elif decision == 'true':
            target_dir = REAL_NG_DIR
        elif decision == 'skip_error':
            # 如果是坏文件，可以移到一个专门的 error 目录，或者直接跳过
            # 这里为了简单，我们把它也作为 True NG 处理，或者直接忽略在代码里处理
            pass

        if target_dir:
            try:
                # 释放图片文件占用（Tkinter 有时会占用文件句柄，重新加载通常会解决，但这里的逻辑是在加载新图前移动）
                # 注意：self.tk_image 只是引用，Image.open 默认是 lazy 的，但在 thumbnail 后通常会关闭
                # 如果遇到 PermissionError，可能需要显式 close，但 PIL 通常处理得很好
                
                shutil.move(self.current_image_path, os.path.join(target_dir, filename))
                print(f"moved {filename} -> {os.path.basename(target_dir)}")
            except Exception as e:
                # 遇到文件占用或其他问题
                messagebox.showerror("移动失败", f"无法移动文件:\n{e}")
                return # 不继续加载下一张，让用户处理

        # 加载下一张
        self.load_next_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ManualReviewApp(root)
    root.mainloop()

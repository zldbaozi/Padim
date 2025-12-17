import os
from PIL import Image

def process_images(source_folder, target_folder, target_size):
    """
    批量处理图像：缩放、转换为灰度图并按数字顺序重命名
    
    参数:
    source_folder: 源图像文件夹路径
    target_folder: 目标保存文件夹路径  
    target_size: 目标尺寸元组 (宽度, 高度)
    """
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"创建目标文件夹: {target_folder}")
    
    # 获取源文件夹中所有图像文件
    image_files = []
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            image_files.append(filename)
    
    # 按文件名排序
    image_files.sort()
    
    # 处理每个图像文件
    success_count = 0
    for i, filename in enumerate(image_files, 1):  # 从1开始计数
        try:
            # 构建完整文件路径
            src_path = os.path.join(source_folder, filename)
            
            # 打开并处理图像
            with Image.open(src_path) as img:
                # 转换为灰度图
                #gray_img = img.convert('L')
                
                # 缩放图像
                #resized_img = gray_img.resize(target_size, Image.LANCZOS)
                #resized_img = img
                resized_img = img.resize(target_size, Image.LANCZOS)
                
                # 生成新文件名（简单的数字命名，从1开始）
                new_filename = f"{i}.bmp"  # 直接使用数字，如1.jpg, 2.jpg等
                dst_path = os.path.join(target_folder, new_filename)
                
                # 保存图像
                resized_img.save(dst_path, "BMP")
                print(f"已处理: {filename} -> {new_filename}")
                success_count += 1
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"\n处理完成! 成功处理 {success_count} 个文件，保存至 {target_folder}")

# 设置参数
source_folder = "datasets/2"      # 源文件夹路径
target_folder = "datasets/NG"     # 目标文件夹路径
target_size = (64, 64)               # 请将n和m替换为实际的宽度和高度值

# 执行处理
if __name__ == "__main__":
    # 例如：target_size = (300, 200)  # 宽度300像素，高度200像素
    process_images(source_folder, target_folder, target_size)
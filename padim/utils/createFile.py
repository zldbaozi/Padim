# prepare_test_structure.py
import os
import shutil
import random

def prepare_test_structure():
    """准备标准的测试目录结构"""
    train_dir = "./data/train/normal"
    test_dir = "./data/test"
    
    # 创建测试目录结构
    normal_test_dir = os.path.join(test_dir, "normal")
    abnormal_test_dir = os.path.join(test_dir, "abnormal")
    os.makedirs(normal_test_dir, exist_ok=True)
    os.makedirs(abnormal_test_dir, exist_ok=True)
    
    # 获取训练图像
    train_images = [f for f in os.listdir(train_dir) if f.endswith('.bmp')]
    
    if not train_images:
        print("❌ 训练目录中没有找到BMP图像")
        return
    
    # 选择一些图像作为正常测试样本
    normal_test_images = random.sample(train_images, min(3, len(train_images)))
    
    # 选择一些图像作为异常测试样本（在实际应用中，这些应该是真正的异常图像）
    # 这里暂时也用正常图像，但你之后应该替换为真正的异常图像
    abnormal_test_images = random.sample([img for img in train_images if img not in normal_test_images], 
                                       min(2, len(train_images) - len(normal_test_images)))
    
    print("准备测试数据...")
    
    # 复制正常测试样本
    for img in normal_test_images:
        src = os.path.join(train_dir, img)
        dst = os.path.join(normal_test_dir, img)
        shutil.copy2(src, dst)
        print(f"✅ 正常测试: {img}")
    
    # 复制异常测试样本（实际使用时请替换为真正的异常图像）
    for img in abnormal_test_images:
        src = os.path.join(train_dir, img)
        dst = os.path.join(abnormal_test_dir, img)
        shutil.copy2(src, dst)
        print(f"⚠️  异常测试: {img} (注意: 这实际上是正常图像，请替换为真实异常图像)")
    
    print(f"\n📁 测试目录结构:")
    print(f"   {normal_test_dir}/ - {len(normal_test_images)} 张正常测试图像")
    print(f"   {abnormal_test_dir}/ - {len(abnormal_test_images)} 张异常测试图像")
    print(f"\n💡 提示: 请将 {abnormal_test_dir}/ 中的图像替换为真实的异常图像")

if __name__ == "__main__":
    prepare_test_structure()
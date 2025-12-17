import argparse
import sys
import os

# 添加路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    """直接训练模型，避免参数解析"""
    from scripts.train import main as train_main
    
    # 硬编码训练参数
    class Args:
        data_path = "C:\\Users\\mento\\Desktop\\data2\\OK"
        output_dir = "./saved_models"
        batch_size = 8
        image_size = 112
        reduce_dims = 100
    
    print("开始训练PaDiM模型...")
    print(f"数据路径: {Args.data_path}")
    print(f"输出目录: {Args.output_dir}")
    
    # 直接调用训练函数
    train_main(Args)

def detect_anomaly():
    """直接进行异常检测，避免参数解析"""
    from scripts.inference import main as inference_main
    
    class Args:
        model_dir = "./saved_models"
        test_data = "C:\\Users\\mento\\Desktop\\data2\\NG"
        threshold = 3.0
        batch_size = 1
        image_size = 112
        save_heatmap = True  # 开启热力图功能！
        save_results = True
        save_dir = "./detection_results"

    print("开始异常检测...")
    print(f"模型目录: {Args.model_dir}")
    print(f"测试数据: {Args.test_data}")
    print(f"检测阈值: {Args.threshold}")
    print(f"🔥 热力图保存: 已开启")
    print(f"📁 结果保存: {Args.save_dir}")
    print(f"🔥 热力图保存: ./detection_heatmaps/")
    
    inference_main(Args)

def main():
    """主函数 - 提供简单的菜单选择"""
    print("=" * 50)
    print("PaDiM 异常检测系统")
    print("=" * 50)
    print("1. 训练模型")
    print("2. 异常检测")
    print("3. 退出")
    print("=" * 50)
    
    while True:
        choice = input("请选择操作 (1/2/3): ").strip()
        
        if choice == "1":
            train_model()
            break
        elif choice == "2":
            detect_anomaly()
            break
        elif choice == "3":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
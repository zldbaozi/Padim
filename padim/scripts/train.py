import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(args):
    """
    修改后的训练函数，直接接收参数对象
    """
    from models.padim_trainer import PaDiMTrainer
    from utils.data_loader import create_data_loader

    print("创建数据加载器...")
    train_loader = create_data_loader(
        args.data_path, 
        batch_size=args.batch_size, 
        image_size=args.image_size,
        is_train=True
    )
    
    print("初始化训练器...")
    trainer = PaDiMTrainer()
    
    print("开始训练...")
    trainer.fit(train_loader, reduce_dims=args.reduce_dims)
    
    print("保存模型...")
    trainer.save_model(args.output_dir)
    
    print("训练完成！")

# 保留原有的命令行接口，便于单独运行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练PaDiM模型')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./saved_models')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=784)     #经过特征提取后的图片分辨率降为28*28=784
    parser.add_argument('--reduce_dims', type=int, default=100)
    
    cli_args = parser.parse_args()
    main(cli_args)
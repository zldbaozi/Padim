import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import copy
import argparse
import time
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

# 新增：保持长宽比的填充类
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

def convert_to_tensor_if_needed(obj):
    if not isinstance(obj, torch.Tensor):
        return torch.tensor(obj)
    return obj

class ReviewModelTrainer:
    def __init__(self, data_root, save_dir, num_epochs=25, batch_size=16, resume=True):
        """
        初始化训练器
        :param data_root: 数据根目录，结构应为 root/real_ng 和 root/false_ng
        :param save_dir: 模型保存目录
        :param resume: 是否加载之前的模型继续训练 (增量学习的关键)
        """
        self.data_root = data_root
        self.save_dir = save_dir
        self.model_save_path = os.path.join(save_dir, "resnet18_review.pth")
        self.onnx_save_path = os.path.join(save_dir, "resnet18_review.onnx")
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.resume = resume
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['false_ng', 'real_ng'] # 0: 假报警(良品), 1: 真缺陷

        os.makedirs(save_dir, exist_ok=True)

    def _get_start_model(self):
        # 1. 定义骨架
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 🔥 优化1：解冻部分层 (Fine-tuning)
        # ⚠️ 数据量少(仅200张)时，解冻太多层会导致过拟合或无法收敛。
        # 改为：只解冻 Layer4 和 FC，锁住 Layer1-3。
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        num_ftrs = model.fc.in_features
        # 🔥 优化2：增强分类头 (Dropout + Linear)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
        
        # 2. 增量学习逻辑
        if self.resume and os.path.exists(self.model_save_path):
            print(f"🔄 发现旧模型，正在加载以进行增量训练: {self.model_save_path}")
            model.load_state_dict(torch.load(self.model_save_path))
        else:
            print("🆕 未找到旧模型或选择全新训练，使用 ImageNet 预训练权重。")
        
        return model.to(self.device)



    def train(self):
        print(f"[System] 使用设备: {self.device}")
        
        # --- 1. 数据增强 ---
        # 🔧 调整: 224 可能导致微小缺陷丢失，提升分辨率至 448
        target_size = (448, 448) 
        
        # 🔥 优化3：简化增强策略，保留最稳健的翻转
        data_transforms = {
            'train': transforms.Compose([
                SquarePad(),
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # 移除旋转和颜色抖动，先确保基础特征能被提取
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                SquarePad(),
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # --- 2. 加载数据 ---
        if not os.path.exists(self.data_root):
            print(f"❌ 错误: 数据目录不存在 {self.data_root}")
            return

        full_dataset = datasets.ImageFolder(self.data_root, data_transforms['train'])
        
        class_to_idx = full_dataset.class_to_idx
        print(f"🗂️ 类别映射: {class_to_idx}")
        
        # 自动划分
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        if total_size < 5:
            print("❌ 数据太少，无法训练。")
            return

        # 🔧 修改：显式打乱并划分，确保随机性
        print("🔀 正在执行全局随机打乱 (Manual Shuffle)...")
        indices = torch.randperm(total_size).tolist()
        train_indices_list = indices[:train_size]
        val_indices_list = indices[train_size:]
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices_list)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices_list)
        
        # === 🔥 关键修复：使用 WeightedRandomSampler 解决不平衡 ===
        print("⚖️ 正在计算采样权重 (WeightedRandomSampler)...")
        # 获取训练集中的所有标签
        train_indices = train_dataset.indices
        train_labels = [full_dataset.targets[i] for i in train_indices]
        
        count_0 = train_labels.count(0) # false_ng
        count_1 = train_labels.count(1) # real_ng
        print(f"   - 假报警(False NG): {count_0} 张")
        print(f"   - 真缺陷(Real NG) : {count_1} 张")
        
        sampler = None
        if count_0 > 0 and count_1 > 0:
            # 计算样本权重：数量少的样本权重高
            weight_0 = 1.0 / count_0
            weight_1 = 1.0 / count_1
            samples_weights = [weight_0 if label == 0 else weight_1 for label in train_labels]
            sampler = WeightedRandomSampler(convert_to_tensor_if_needed(samples_weights), num_samples=len(samples_weights), replacement=True)
            print(f"   - ✅ 已启用过采样策略，每个 Batch 将包含均衡的正负样本")
        else:
            print("⚠️ 警告: 即使划分后也就是某一类样本为0，无法计算权重。")

        # 注意：使用 sampler 时，shuffle 必须为 False
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=0),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        }
        dataset_sizes = {'train': train_size, 'val': val_size}
        print(f"📊 数据量: 训练集 {train_size} 张, 验证集 {val_size} 张")

        # --- 3. 准备模型 ---
        model = self._get_start_model()

        # 🔥 不需要再使用 weight 参数了，因为 Sampler 已经解决了平衡问题
        criterion = nn.CrossEntropyLoss()
        
        # 🔥 优化4：改用 Adam 优化器，对不同量级的梯度更鲁棒
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        # 🔥 优化5：StepLR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        # --- 4. 训练循环 ---
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        

        start_time = time.time()

        for epoch in range(self.num_epochs):            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                # 打印详细日志，监控是否存在过拟合/欠拟合
                print(f'Epoch {epoch+1}/{self.num_epochs} | {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 只有 validation 更好时才更新最佳模型
                if phase == 'val':
                    # ⚠️ 简单打印一下最后的预测情况，用于目测分布（不计算完整 CM 以免拖慢）
                    # 下一次如果 Acc 还是上不去，建议专门写一个 eval 脚本
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - start_time
        print(f'✅ 训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'🏆 最佳验证集准确率: {best_acc:.4f}')

        # --- 5. 保存模型 ---
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), self.model_save_path)
        print(f"💾 PyTorch 模型已保存: {self.model_save_path}")

        # --- 6. 导出 ONNX (供 C++ 使用) ---
        self._export_onnx(model)

    def _export_onnx(self, model):
        model.eval()
        # ⚠️ 注意：ONNX 导出时的输入分辨率必须与训练时一致 (448x448)
        dummy_input = torch.randn(1, 3, 448, 448).to(self.device)
        try:
            print("[torch.onnx] Starting ONNX export...")
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_save_path,
                input_names=['input'],
                output_names=['output'],
                # dynamic_shapes=False, # ❌ 这是一个错误的参数，之前导致了导出失败
                opset_version=18  # 使用更通用的 opset 11
            )
            print(f"📦 ONNX 模型已导出: {self.onnx_save_path}")
        except Exception as e:
            print("❌ ONNX 导出失败: 以下是详细错误信息")
            print(e)
            print("\n建议检查以下内容:")
            print("1. 模型的输入/输出是否符合 ONNX 要求。")
            print("2. 是否有不支持 ONNX 的层或操作。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================= 整合后的路径配置 =================
    # 默认指向 AOI_Integrated/data/dataset_review/train
    default_data_dir = r'e:\Code\AOI_Integrated\data\dataset_review\train'
    # 默认模型保存到 AOI_Integrated/models/resnet
    default_save_dir = r'e:\Code\AOI_Integrated\models\resnet'

    
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='训练数据根目录')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    args = parser.parse_args()

    trainer = ReviewModelTrainer(args.data_dir, args.save_dir, num_epochs=args.epochs)
    trainer.train()


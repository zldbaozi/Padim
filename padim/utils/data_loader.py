import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        for img_name in os.listdir(data_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg','.bmp','BMP')):
                self.image_paths.append(os.path.join(data_dir, img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, os.path.basename(img_path)

def get_transforms(image_size=112, is_train=True):
    # ImageNet 标准化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if is_train:
        return transforms.Compose([
            # 1. 先缩放到 112
            transforms.Resize((image_size, image_size)),
            
            # 2. 转为 Tensor 并归一化
            # 注意：先归一化，这样后面的 fill=0 填充的就是真正的“零信息”区域
            # 如果后归一化，fill=0 会被归一化成非零值（比如 -2.11），这也没问题，但先归一化逻辑更清晰
            transforms.ToTensor(),
            normalize,         
           
        ])
    else:
        # 测试集：保持一致
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
def create_data_loader(data_dir, batch_size=8, image_size=112, is_train=True, shuffle=True):
    # ... (保持不变) ...
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    normal_dir = os.path.join(data_dir, 'normal')
    abnormal_dir = os.path.join(data_dir, 'abnormal')
    
    if os.path.exists(normal_dir) and os.path.exists(abnormal_dir):
        print(f"检测到标准目录结构: {data_dir}/normal 和 {data_dir}/abnormal")
        dataset_normal = CustomDataset(normal_dir, transform=get_transforms(image_size, is_train))
        dataset_abnormal = CustomDataset(abnormal_dir, transform=get_transforms(image_size, is_train))
        
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([dataset_normal, dataset_abnormal])
        print(f"合并数据集: {len(dataset_normal)} 正常 + {len(dataset_abnormal)} 异常 = {len(dataset)} 总图像")
        
    else:
        print(f"使用单一目录: {data_dir}")
        transform = get_transforms(image_size, is_train)
        dataset = CustomDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"在目录 {data_dir} 中没有找到任何图像文件！")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
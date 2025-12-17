import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    使用预训练的resnet18提取多尺度特征
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 加载预训练模型
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 获取不同层的输出
        self.layer1 = nn.Sequential(*list(model.children())[:4])
        self.layer2 = model.layer1
        self.layer3 = model.layer2
        self.layer4 = model.layer3
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        return [x2, x3, x4]
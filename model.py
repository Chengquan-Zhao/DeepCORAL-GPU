"""AlexNet + 31 类 fc8（论文 Fig.1）。

论文用的是 Caffe 的 BVLC Reference CaffeNet；这里用 torchvision 的 ImageNet 预训练 AlexNet
代替（结构几乎一致，区别是没有 LRN、通道数略有不同），这是 PyTorch 复现中的通行做法。
"""
import torch.nn as nn
from torchvision.models import AlexNet_Weights, alexnet


class DeepCORAL(nn.Module):
    def __init__(self, num_classes: int = 31, pretrained: bool = True):
        super().__init__()
        net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.avgpool = net.avgpool
        # fc6、fc7 保留预训练参数（classifier[6] 是原来的 1000 类 fc8，丢弃）
        self.fc67 = nn.Sequential(*list(net.classifier.children())[:6])
        # fc8：维度 = 类别数，权重 ~ N(0, 0.005)，偏置为 0（论文第 4 节）
        self.fc8 = nn.Linear(4096, num_classes)
        nn.init.normal_(self.fc8.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.fc8.bias)

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        return self.fc8(self.fc67(x))   # fc8 的输出既用于分类损失，也用于 CORAL 损失

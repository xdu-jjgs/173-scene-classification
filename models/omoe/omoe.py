import torch
import torch.nn as nn

from typing import List

from models.omoe.gate import Gate
from models.omoe.classifier import ImageClassifier


class OMOE(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(OMOE, self).__init__()
        # backbone输入通道数
        pool_size = 4
        self.num_channels = experts[0].in_channels
        self.experts = nn.ModuleList(experts)
        self.gates = Gate(self.num_channels * pool_size * pool_size, len(experts))
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.gap = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    def forward(self, x):
        b, c, h, w = x.size()
        experts_features = [i(x) for i in self.experts]
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)
        x_gap = self.gap(x).view(b, -1)

        task_weight = self.gates(x_gap)[-1].softmax(dim=1).unsqueeze(1)
        # print(task_weight.size(), experts_features.size())
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)
        out = self.classifier(features)
        # print(features.size(), out.size())
        return out, task_weight

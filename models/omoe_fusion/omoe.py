import torch
import torch.nn as nn

from typing import List, Tuple

from models.modules.gate import Gate
from models.modules.classifier import ImageClassifier


class OMOEFusion(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(OMOEFusion, self).__init__()
        # backbone输入通道数
        pool_size = 6
        self.num_channels = sum(i.in_channels for i in experts)
        self.experts = nn.ModuleList(experts)
        self.gates = Gate(self.num_channels * pool_size * pool_size, len(experts))
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.gap = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    def forward(self, xs: Tuple):
        experts_features = []
        for x, expert in zip(xs, self.experts):
            experts_features.append(expert(x))
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)
        x_concat = torch.concat(xs, dim=1)  #NCHW
        b, c, h, w = x_concat.size()
        x_gap = self.gap(x_concat).view(b, -1)

        task_weight = self.gates(x_gap).softmax(dim=1).unsqueeze(1)
        # print(task_weight.size(), experts_features.size())
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)
        out = self.classifier(features)
        # print(features.size(), out.size())
        return out, task_weight

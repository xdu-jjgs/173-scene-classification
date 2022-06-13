import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pooling_size: Tuple[int] = (1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList()
        assert in_channels % len(pooling_size) == 0
        out_channels = in_channels // len(pooling_size)
        for index, size in enumerate(pooling_size):
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, inp):
        size = inp.size()
        oup = []
        for stage in self.stages:
            x = stage(inp)
            # NCHW
            x = F.upsample_bilinear(x, size=size[2:])
            oup.append(x)
        oup = torch.cat((inp, *oup), dim=1)
        return oup

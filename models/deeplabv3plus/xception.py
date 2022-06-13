import re
import torch.nn as nn
from typing import Tuple

from models.utils.init import initialize_weights
from models.utils.download import download_models


class SeparableConv2d(nn.Module):
    # output_stride = 16
    # Base on the modified version (https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py)
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: int, padding: int,
                 dilation=1, bias: bool = False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                               groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1, padding: int = 1,
                 start_with_relu: bool = True):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.skip = None
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        self.rep = [
            self.relu,
            SeparableConv2d(in_channels, mid_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            self.relu,
            SeparableConv2d(mid_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu,
            SeparableConv2d(out_channels, out_channels, (3, 3), stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
        ]
        if not start_with_relu:
            self.rep = self.rep[1:]
        self.rep = nn.Sequential(*self.rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x


class ModifiedAlignedXception(nn.Module):
    # output stride = 16
    def __init__(self, in_channels: int, pretrained=True):
        super(ModifiedAlignedXception, self).__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 128, stride=2, padding=1, start_with_relu=False)
        self.block2 = Block(128, 256, 256, stride=2, padding=1)
        self.block3 = Block(256, 728, 728, stride=2, padding=1)

        # Middle flow
        self.block4 = Block(728, 728, 728)
        self.block5 = Block(728, 728, 728)
        self.block6 = Block(728, 728, 728)
        self.block7 = Block(728, 728, 728)
        self.block8 = Block(728, 728, 728)

        self.block9 = Block(728, 728, 728)
        self.block10 = Block(728, 728, 728)
        self.block11 = Block(728, 728, 728)
        self.block12 = Block(728, 728, 728)
        self.block13 = Block(728, 728, 728)

        self.block14 = Block(728, 728, 728)
        self.block15 = Block(728, 728, 728)
        self.block16 = Block(728, 728, 728)
        self.block17 = Block(728, 728, 728)
        self.block18 = Block(728, 728, 728)

        self.block19 = Block(728, 728, 728)

        # Exit flow
        # self.block20 = Block(728, 728, 1024, stride=2, padding=1)
        self.block20 = Block(728, 728, 1024)

        self.conv3 = SeparableConv2d(1024, 1536, (3, 3), stride=1, padding=1, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 1536, (3, 3), stride=1, padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d(1536, 2048, (3, 3), stride=1, padding=1, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(2048)

        if pretrained:
            self._load_pretrained_model()
        else:
            initialize_weights(self)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_features = x
        x = self.block2(x)
        # low_level_features = x
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)

        x = self.block19(x)

        # Exit flow
        x = self.block20(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_features

    def _load_pretrained_model(self):
        pretrained_dict = download_models('xception')
        state_dict = self.state_dict()
        state_dict_new = {}
        '''
        Remove list:
        ①BN的num_batches_tracked参数
        ②depthwise后的BN
        ③block1的最后一层SepConv和BN
        ④block2、3、20最后一层的SepConv和BN
        ⑤block12-19
        '''
        state_dict_ll = list(
            filter(lambda x: not re.findall(r'batches_tracked|'
                                            r'\.bn1|'
                                            r'block1\.\S+[67]|'
                                            r'block[23]0?\.\S+\.[78]|'
                                            r'block1[2-9]'
                                            , x),
                   state_dict.keys()))
        # print(state_dict_ll)
        # print(list(pretrained_dict.keys()))
        '''
        Set list
        ①pointwise(add dims)->pointwise
        ②block11(pretrained)->block11-19
        ③block12(pretrained)->block20
        ④bn3(pretrained)->bn3,bn4(1536)
        ⑤bn4(pretrained)->bn5(2048)
        ⑥conv4(pretrained)->conv5(1536-2048) & null->conv4
        ⑦fc(pretrained)->null
        '''
        for key, val in pretrained_dict.items():
            if 'pointwise' in key:
                # Size([128, 64]) -> Size([128, 64, 1, 1])
                val = val.unsqueeze(-1).unsqueeze(-1)
            if key.startswith('block11'):
                state_dict_new[key] = val
                state_dict_new[key.replace('block11', 'block12')] = val
                state_dict_new[key.replace('block11', 'block13')] = val
                state_dict_new[key.replace('block11', 'block14')] = val
                state_dict_new[key.replace('block11', 'block15')] = val
                state_dict_new[key.replace('block11', 'block16')] = val
                state_dict_new[key.replace('block11', 'block17')] = val
                state_dict_new[key.replace('block11', 'block18')] = val
                state_dict_new[key.replace('block11', 'block19')] = val
            elif key.startswith('block12'):
                state_dict_new[key.replace('block12', 'block20')] = val
            elif key.startswith('bn3'):
                state_dict_new[key] = val
                state_dict_new[key.replace('bn3', 'bn4')] = val
            elif key.startswith('bn4'):
                state_dict_new[key.replace('bn4', 'bn5')] = val
            elif key.startswith('conv4'):
                state_dict_new[key.replace('conv4', 'conv5')] = val
            elif key.startswith('fc'):
                continue
            elif key.startswith('conv1') and self.conv1.in_channels != 3:
                continue
            else:
                state_dict_new[key] = val

        state_dict.update(state_dict_new)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    xception = ModifiedAlignedXception(3)

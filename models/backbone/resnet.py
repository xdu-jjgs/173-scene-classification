import torch.nn as nn
import torchvision.models as models

from models.utils.download import load_pretrained_models


class ResNet(nn.Module):
    def __init__(self, depth, in_channels, pretrained=True, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.model_name = 'resnet{}'.format(depth)
        model = getattr(models, self.model_name)(replace_stride_with_dilation=replace_stride_with_dilation)
        depth2channels = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
        }
        self.out_channels = depth2channels[depth]

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, 7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import ResNet
from .decoder import DeepLabV3Decoder
from models.utils.init import initialize_weights


class DeepLabV3ResNet(nn.Module):
    def __init__(self, depth, in_channels, num_classes):
        super(DeepLabV3ResNet, self).__init__()
        self.encoder = ResNet(depth, in_channels)
        self.mid_channels = self.encoder.out_channels
        self.decoder = DeepLabV3Decoder(self.mid_channels, 512, num_classes)
        initialize_weights(self.decoder)

    def forward(self, x):
        _, _, h, w = x.shape

        x = self.encoder(x)[-1]
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class DeepLabV3ResNet18(DeepLabV3ResNet):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3ResNet18, self).__init__(18, in_channels, num_classes)


class DeepLabV3ResNet34(DeepLabV3ResNet):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3ResNet34, self).__init__(34, in_channels, num_classes)


class DeepLabV3ResNet50(DeepLabV3ResNet):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3ResNet50, self).__init__(50, in_channels, num_classes)


class DeepLabV3ResNet101(DeepLabV3ResNet):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3ResNet101, self).__init__(101, in_channels, num_classes)

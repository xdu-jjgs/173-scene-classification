import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import ResNet
from models.utils.init import initialize_weights
from models.pspnet.ppm import PyramidPoolingModule


class PSPNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int = 101, pretrained: bool = True):
        super(PSPNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1

        self.resnet = ResNet(depth=depth, in_channels=in_channels, pretrained=pretrained,
                             replace_stride_with_dilation=[False, True, True])
        mid_channels = self.resnet.out_channels
        self.pyramid_pooling_module = PyramidPoolingModule(mid_channels)
        self.final = nn.Sequential(
            nn.Conv2d(mid_channels * 2, 512, kernel_size=(1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.out_channels, 1, stride=1, padding=0)
        )

        initialize_weights(self.pyramid_pooling_module)
        initialize_weights(self.final)

    def forward(self, x):
        size = x.size()
        x = self.resnet(x)[-1]
        x = self.pyramid_pooling_module(x)
        x = self.final(x)
        x = F.upsample_bilinear(x, size=size[2:])
        return x


if __name__ == '__main__':
    print(PSPNet(10, 10))

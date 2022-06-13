import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import ResNet
from models.utils.init import initialize_weights
from models.danet.attention_modules import PositionAttentionModule, ChannelAttentionModule


class DANet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(DANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1

        self.resnet = ResNet(101, in_channels, replace_stride_with_dilation=[False, True, True])
        mid_channels = self.resnet.out_channels
        assert mid_channels % 4 == 0
        reduction_channels = mid_channels // 4

        self.relu = nn.ReLU(inplace=True)
        self.pam_convbnrelu1 = nn.Sequential(
            nn.Conv2d(mid_channels, reduction_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            self.relu
        )
        self.position_attention_module = PositionAttentionModule(reduction_channels)
        self.pam_convbnrelu2 = nn.Sequential(
            nn.Conv2d(reduction_channels, reduction_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            self.relu
        )

        self.cam_convbnrelu1 = nn.Sequential(
            nn.Conv2d(mid_channels, reduction_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            self.relu
        )
        self.channel_attention_module = ChannelAttentionModule()
        self.cam_convbnrelu2 = nn.Sequential(
            nn.Conv2d(reduction_channels, reduction_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            self.relu
        )

        self.final = nn.Sequential(
            nn.Dropout2d(0.1, inplace=True),
            nn.Conv2d(reduction_channels, self.out_channels, (1, 1), stride=1, padding=0)
        )

        initialize_weights(self.pam_convbnrelu1)
        initialize_weights(self.position_attention_module)
        initialize_weights(self.pam_convbnrelu2)
        initialize_weights(self.cam_convbnrelu1)
        initialize_weights(self.channel_attention_module)
        initialize_weights(self.cam_convbnrelu2)
        initialize_weights(self.final)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.resnet(x)[-1]

        pam = self.pam_convbnrelu1(x)
        pam = self.position_attention_module(pam)
        pam = self.pam_convbnrelu2(pam)

        cam = self.cam_convbnrelu1(x)
        cam = self.channel_attention_module(cam)
        cam = self.cam_convbnrelu2(cam)

        out = pam + cam
        out = self.final(out)
        out = F.upsample_bilinear(out, size=(height, width))
        return out

import torch.nn as nn

from models.backbone.fpn import FPN
from models.backbone.resnet import ResNet
from models.farseg.decoder import FarSegDecoder
from models.farseg.relation_modules import ForegroundSceneModule
from models.utils.init import initialize_weights


class FarSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int = 50):
        super(FarSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1

        # output stride = 32
        self.resnet = ResNet(depth=depth, in_channels=self.in_channels, pretrained=True)
        mid_channels = self.resnet.out_channels
        channel_list = [256, 512, 1024, 2048]
        fs_channels = 256
        self.fpn = FPN(mid_channels, fs_channels, channel_list)
        self.foreground_relation = ForegroundSceneModule(fs_channels, fs_channels, scene_channels=channel_list[-1],
                                                         depth_fpn=len(channel_list))
        decoder_channels = 128
        self.decoder = FarSegDecoder(fs_channels, self.out_channels, mid_channels=decoder_channels)

        # TODO: ADD FA

        initialize_weights(self.fpn)
        initialize_weights(self.foreground_relation)
        initialize_weights(self.decoder)

    def forward(self, x):
        _, _, height, width = x.size()
        xs = self.resnet(x)[1:]

        C5 = xs[-1]
        xs = self.fpn(xs)
        xs = self.foreground_relation(xs, C5)
        x = self.decoder(xs, (height, width))
        return x

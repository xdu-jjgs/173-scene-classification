import torch.nn as nn

from .encoder import UNetEncoder
from .decoder import UNetDecoder
from models.utils.init import initialize_weights


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, with_bn: bool = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.encoder_channels = [64, 128, 256, 512]
        self.center_channels = 1024
        self.decoder_channels = [512, 256, 128, 64]
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1
        self.with_bn = with_bn

        self.encoder = UNetEncoder(self.in_channels, self.encoder_channels, self.with_bn)
        self.center_block = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], self.center_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = UNetDecoder(self.center_channels, self.decoder_channels, self.with_bn)
        self.output = nn.Sequential(
            nn.Conv2d(self.decoder_channels[-1], self.out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        initialize_weights(self.encoder)
        initialize_weights(self.center_block)
        initialize_weights(self.decoder)
        initialize_weights(self.output)

    def forward(self, x):
        x, x_before_pools = self.encoder(x)
        x = self.center_block(x)
        x = self.decoder(x, x_before_pools)
        x = self.output(x)
        return x

import torch
import torch.nn as nn
from typing import List

from models.unet.decoder import UNetDecoder
from models.unet.encoder import UNetEncoder
from models.brrnet.brrnet_atrous_conv import BRRNetAtrousConv
from models.utils.init import initialize_weights


class PredictModule(nn.Module):
    def __init__(self, in_channels: int, encoder_channels: List[int], center_channels: int, decoder_channels: List[int],
                 out_channels: int, with_bn: bool):
        super(PredictModule, self).__init__()

        self.encoder = UNetEncoder(in_channels, encoder_channels, with_bn)
        self.atrous_conv_block = BRRNetAtrousConv(encoder_channels[-1], center_channels)
        self.decoder = UNetDecoder(center_channels, decoder_channels, with_bn)
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Sequential?
        x, x_before_pools = self.encoder(x)
        # print("encoder output {}".format(x.shape))
        x = self.atrous_conv_block(x)
        # print("middle output {}".format(x.shape))
        x = self.decoder(x, x_before_pools)
        # print("decoder output {}".format(x.shape))
        x = self.output(x)
        return x


class ResidualRefinementModule(nn.Module):
    def __init__(self, in_channels: int, center_channels: int, out_channels: int):
        super(ResidualRefinementModule, self).__init__()

        self.atrous_conv_block = BRRNetAtrousConv(in_channels, center_channels)
        self.output = nn.Sequential(
            nn.Conv2d(center_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inp = x
        x = self.atrous_conv_block(x)  # does inp change?
        x = self.output(x)
        x = torch.add(inp, x)
        x = torch.sigmoid(x)
        return x


class BRRNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(BRRNet, self).__init__()
        self.in_channels = in_channels
        self.encoder_channels = [64, 128, 256]
        self.predict_center_channels = 512
        self.decoder_channels = [256, 128, 64]
        self.rrm_center_channels = 64
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1
        self.with_bn = True

        # Predict Module
        self.predict_module = PredictModule(self.in_channels, self.encoder_channels, self.predict_center_channels,
                                            self.decoder_channels, self.out_channels, self.with_bn)
        # Residual Refinement Module
        self.residual_refinement_module = ResidualRefinementModule(self.out_channels, self.rrm_center_channels,
                                                                   self.out_channels)

        initialize_weights(self.predict_module)
        initialize_weights(self.residual_refinement_module)

    def forward(self, x):
        # print("Input {}".format(x.shape))
        x = self.predict_module(x)
        # print("predict module output {}".format(x.shape))
        x = self.residual_refinement_module(x)
        # print("residual module output {}".format(x.shape))
        return x

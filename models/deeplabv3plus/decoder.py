import torch
import torch.nn as nn


class DeeplabV3PlusDecoder(nn.Module):
    # output stride = 16
    def __init__(self, in_channels: int, out_channels: int):
        super(DeeplabV3PlusDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(in_channels, 48, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48),
            self.relu
        )
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.cat_conv = nn.Sequential(
            nn.Conv2d(304, 256, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            self.relu,
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            self.relu,
            nn.Dropout(0.1),

            nn.Conv2d(256, out_channels, (1, 1), stride=1, padding=0),
        )
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x, low_level_features):
        low_level_features = self.low_level_conv(low_level_features)
        x = self.upsample1(x)
        # print("output of encoder: {}".format(x.shape))
        # print("low level: {}".format(low_level_features.shape))
        x = torch.cat((x, low_level_features), dim=1)
        x = self.cat_conv(x)
        x = self.upsample2(x)
        return x

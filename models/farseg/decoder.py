import torch
import torch.nn as nn
import torch.nn.functional as F


class FarSegDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(FarSegDecoder, self).__init__()

        mid_channels = mid_channels if mid_channels else in_channels

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(mid_channels, out_channels, (1, 1), stride=1, padding=0)

    def forward(self, xs, output_size):
        height, width = output_size
        pre = None
        for x in xs:
            x = self.convbnrelu(x)
            x = F.upsample_bilinear(x, size=(height // 4, width // 4))
            if pre is not None:
                oup = x + pre
            pre = x
        oup = oup / len(xs)
        oup = self.final(oup)
        oup = F.upsample_bilinear(oup, size=output_size)
        return oup

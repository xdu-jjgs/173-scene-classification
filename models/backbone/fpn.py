import torch.nn as nn
import torch.nn.functional as F

from typing import List


class FPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channel_list: List[int]):
        super(FPN, self).__init__()

        self.in_channels = in_channels

        assert len(channel_list) == 4
        # from high solution to low solution
        self.merge_blocks = nn.ModuleList([
            nn.Conv2d(channel, out_channels, (1, 1), stride=1, padding=0)
            for channel in channel_list
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1)
            for _ in channel_list
        ])

    def forward(self, xs):
        # from high solution to low solution(P2-P5)
        res = []

        # from low solution to high solution(C5-C2)
        for index, x in enumerate(xs[::-1], 1):
            x = self.merge_blocks[-1 * index](x)
            if res:
                x += F.upsample_bilinear(res[0], scale_factor=2)
            x = self.decoder_blocks[-1 * index](x)
            res.insert(0, x)
        return tuple(res)

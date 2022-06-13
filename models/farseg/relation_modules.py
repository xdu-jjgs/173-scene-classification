import torch.nn as nn
import torch.nn.functional as F


class ForegroundSceneModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scene_channels: int, depth_fpn: int):
        super(ForegroundSceneModule, self).__init__()

        self.in_channels = in_channels

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(scene_channels, out_channels, (1, 1), stride=1, padding=0),
                self.relu,
                nn.Conv2d(out_channels, out_channels, (1, 1), stride=1, padding=0)
            )
            for _ in range(depth_fpn)
        ])
        self.scale_aware_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                self.relu
            )
            for _ in range(depth_fpn)
        ])
        self.reencode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                self.relu
            )
            for _ in range(depth_fpn)
        ])

    def forward(self, xs, C5):
        # re-weighted
        # 1*1*C
        C6 = self.gap(C5)
        # 1*1*OC
        scene_embedding_vectors = [layer(C6) for layer in self.embedding_layers]
        # H*W*OC
        xs_rescale = [layer(x) for layer, x in zip(self.scale_aware_layers, xs)]
        # H*W*OC dot mul
        relations = [vector * x for vector, x in zip(scene_embedding_vectors, xs_rescale)]

        # re-encoding
        # H*W*OC dot mul
        xs_reencode = [layer(x) for layer, x in zip(self.reencode_layers, xs)]

        oup = [F.sigmoid(relation) * x_reencode for relation, x_reencode in zip(relations, xs_reencode)]
        return oup

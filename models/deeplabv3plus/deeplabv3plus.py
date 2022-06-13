import torch.nn as nn

from .encoder import DeeplabV3PlusEncoder
from .decoder import DeeplabV3PlusDecoder
from models.utils.init import initialize_weights


class DeeplabV3Plus(nn.Module):
    # output stride = 16
    def __init__(self, in_channels: int, num_classes: int):
        super(DeeplabV3Plus, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = 256
        self.low_level_channels = 128
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1

        self.encoder = DeeplabV3PlusEncoder(self.in_channels, self.mid_channels)
        self.decoder = DeeplabV3PlusDecoder(self.low_level_channels, self.out_channels)
        # only init decoder
        initialize_weights(self.decoder)

    def forward(self, x):
        x, low_level_features = self.encoder(x)
        x = self.decoder(x, low_level_features)
        return x

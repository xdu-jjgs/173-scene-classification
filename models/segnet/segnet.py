import torch.nn as nn
from models.segnet.encoder import SegNetEncoder
from models.segnet.decoder import SegNetDecoder
from models.utils.init import initialize_weights


class SegNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes if num_classes > 2 else 1

        self.encoder = SegNetEncoder(self.in_channels)
        self.decoder = SegNetDecoder(self.out_channels)

        # only init decoder
        initialize_weights(self.decoder)

    def forward(self, x):
        x, max_pool_indexes = self.encoder(x)
        x = self.decoder(x, max_pool_indexes)
        return x

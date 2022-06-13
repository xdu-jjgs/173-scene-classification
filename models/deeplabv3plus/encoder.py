import torch.nn as nn
from .xception import ModifiedAlignedXception
from models.deeplabv3.aspp import ASPP
from models.utils.init import initialize_weights


class DeeplabV3PlusEncoder(nn.Module):
    # output stride = 16
    def __init__(self, in_channels: int, out_channels: int):
        super(DeeplabV3PlusEncoder, self).__init__()
        self.xception = ModifiedAlignedXception(in_channels)
        self.aspp = ASPP(2048, out_channels, dilations=(6, 12, 18))
        # only init ASPP
        initialize_weights(self.aspp)

    def forward(self, x):
        x, low_level_features = self.xception(x)
        x = self.aspp(x)
        return x, low_level_features

import torch.nn as nn
import pretrainedmodels

from models.utils.download import load_pretrained_models


class Xception(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pretrained=True):
        super(Xception, self).__init__()
        self.model = pretrainedmodels.__dict__['xception'](pretrained=None)
        if pretrained:
            self.model = load_pretrained_models(self.model, 'xception')
        self.model.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.model.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

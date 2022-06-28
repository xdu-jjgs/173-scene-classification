import torch.nn as nn
import torchvision.models as models

from models.utils.download import load_pretrained_models


class VGG(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=True, with_bn: bool = True):
        super(VGG, self).__init__()
        self.model_name = 'vgg{}'.format(depth)
        self.model_name += '_bn' if with_bn else ''
        self.model = getattr(models, self.model_name)(num_classes=num_classes)
        if pretrained:
            self.model = load_pretrained_models(self.model, self.model_name)
        self.model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.model(x)
        return x

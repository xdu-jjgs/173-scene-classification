import torch.nn as nn
import torchvision.models as models

from models.utils.download import load_pretrained_models


class DenseNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=True):
        super(DenseNet, self).__init__()
        self.model_name = 'densenet{}'.format(depth)
        model = getattr(models, self.model_name)(num_classes=num_classes)

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
        model.features['conv0'] = nn.Conv2d(3, model.features['conv0'].out_channels, kernel_size=7, stride=2, padding=3,
                                            bias=False)

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

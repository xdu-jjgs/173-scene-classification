import torch.nn as nn
import torchvision.models as models

from models.utils.download import load_pretrained_models


class DenseNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=True):
        super(DenseNet, self).__init__()
        self.model_name = 'densenet{}'.format(depth)
        model = getattr(models, self.model_name)(num_init_features=in_channels, num_classes=num_classes)

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
        # model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, 7, stride=2, padding=3, bias=False)

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

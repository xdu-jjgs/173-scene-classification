import torch.nn as nn
import torchvision.models as tv_models

from models.utils.init import initialize_weights
from models.utils.download import load_pretrained_models


class SegNetEncoder(nn.Module):
    def __init__(self, in_channels: int, pretrained: bool = True):
        super(SegNetEncoder, self).__init__()
        self.vgg16_bn = tv_models.vgg16_bn()
        if pretrained:
            self.vgg16_bn = load_pretrained_models(self.vgg16_bn, 'vgg16_bn')
        else:
            initialize_weights(self.vgg16_bn)
        self._replace_max_pool_layer(in_channels)
        # print(*self.vgg16, sep='\n')

    def forward(self, x):
        max_pool_indexes = []
        for module in self.vgg16_bn:
            if isinstance(module, nn.MaxPool2d):
                x, index = module(x)
                max_pool_indexes.append(index)
            else:
                x = module(x)
        return x, max_pool_indexes

    def _replace_max_pool_layer(self, in_channels: int):
        vgg_features_layers = self.vgg16_bn.features.children()
        vgg_new = []
        # Change MaxPool and Conv1
        for module in vgg_features_layers:
            if isinstance(module, nn.MaxPool2d):
                vgg_new[-1] = nn.Sequential(*vgg_new[-1])
                vgg_new.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            # else if empty
            elif not vgg_new:
                # in_channels is 3
                if in_channels == module.in_channels:
                    vgg_new.append([module])
                else:
                    vgg_new.append([nn.Conv2d(in_channels, 64, module.kernel_size,
                                              stride=module.stride, padding=module.padding)])
            # the last is MaxPool2d/MaxUnpool2d
            elif isinstance(vgg_new[-1], (nn.MaxPool2d, nn.MaxUnpool2d)):
                vgg_new.append([module])
            else:
                vgg_new[-1].append(module)
        self.vgg16_bn = nn.ModuleList(vgg_new)


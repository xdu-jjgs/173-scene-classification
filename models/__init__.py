from configs import CFG
from .resnet import ResNet
from .desnsenet import DenseNet
from .xception import Xception
from .vgg import VGG


def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'resnet18':
        return ResNet(num_channels, num_classes, 18)
    elif CFG.MODEL.NAME == 'resnet34':
        return ResNet(num_channels, num_classes, 34)
    elif CFG.MODEL.NAME == 'resnet50':
        return ResNet(num_channels, num_classes, 50)
    elif CFG.MODEL.NAME == 'resnet101':
        return ResNet(num_channels, num_classes, 101)
    elif CFG.MODEL.NAME == 'resnet152':
        return ResNet(num_channels, num_classes, 152)
    elif CFG.MODEL.NAME == 'densenet121':
        return DenseNet(num_channels, num_classes, 121)
    elif CFG.MODEL.NAME == 'densenet161':
        return DenseNet(num_channels, num_classes, 161)
    elif CFG.MODEL.NAME == 'densenet169':
        return DenseNet(num_channels, num_classes, 169)
    elif CFG.MODEL.NAME == 'densenet201':
        return DenseNet(num_channels, num_classes, 201)
    elif CFG.MODEL.NAME == 'xception':
        return Xception(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'vgg16':
        return VGG(num_channels, num_classes, depth=16)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))

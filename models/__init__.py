from configs import CFG
from .resnet import ResNet
from .desnsenet import DenseNet
from .xception import Xception
from .vgg import VGG
from .fusenet import FuseNet


def build_model(num_channels, num_classes, model_name: str = CFG.MODEL.NAME):
    if model_name == 'resnet18':
        return ResNet(num_channels, num_classes, 18)
    elif model_name == 'resnet34':
        return ResNet(num_channels, num_classes, 34)
    elif model_name == 'resnet50':
        return ResNet(num_channels, num_classes, 50)
    elif model_name == 'resnet101':
        return ResNet(num_channels, num_classes, 101)
    elif model_name == 'resnet152':
        return ResNet(num_channels, num_classes, 152)
    elif model_name == 'densenet121':
        return DenseNet(num_channels, num_classes, 121)
    elif model_name == 'densenet161':
        return DenseNet(num_channels, num_classes, 161)
    elif model_name == 'densenet169':
        return DenseNet(num_channels, num_classes, 169)
    elif model_name == 'densenet201':
        return DenseNet(num_channels, num_classes, 201)
    elif model_name == 'xception':
        return Xception(num_channels, num_classes)
    elif model_name == 'vgg16':
        return VGG(num_channels, num_classes, depth=16)
    elif '_fusenet' in model_name:
        classifier = build_model(num_channels, num_classes, model_name.split('_')[0])
        return FuseNet(num_channels // 2, classifier)
    raise NotImplementedError('invalid model: {}'.format(model_name))

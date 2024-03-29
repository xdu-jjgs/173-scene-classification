from .vgg import VGG
from .omoe import OMOE
from configs import CFG
from .resnet import ResNet
from .fusenet import FuseNet
from .xception import Xception
from .desnsenet import DenseNet
from .omoe_fusion import OMOEFusion


def build_model(num_channels, num_classes, model_name: str = None, return_features: bool = False):
    if model_name is None:
        model_name = CFG.MODEL.NAME
    if model_name == 'resnet18':
        return ResNet(num_channels, num_classes, 18, return_features=return_features)
    elif model_name == 'resnet34':
        return ResNet(num_channels, num_classes, 34, return_features=return_features)
    elif model_name == 'resnet50':
        return ResNet(num_channels, num_classes, 50, return_features=return_features)
    elif model_name == 'resnet101':
        return ResNet(num_channels, num_classes, 101, return_features=return_features)
    elif model_name == 'resnet152':
        return ResNet(num_channels, num_classes, 152, return_features=return_features)
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
    elif model_name == 'omoe':
        backbone_ = [build_model(num_channels, num_classes, i, return_features=True) for i in CFG.MODEL.EXPERTS]
        return OMOE(num_classes, backbone_)
    elif model_name == 'omoe_fusion':
        backbone_ = [build_model(i, num_classes, j, return_features=True) for i, j in
                     zip(num_channels, CFG.MODEL.EXPERTS)]
        return OMOEFusion(num_classes, backbone_)
    elif '_fusenet' in model_name:
        classifier1 = build_model(num_channels[0], num_classes, model_name.split('_')[0], return_features=True)
        classifier2 = build_model(num_channels[1], num_classes, model_name.split('_')[0], return_features=True)
        return FuseNet(num_classes, classifier1, classifier2)
    # print(model_name, CFG)
    raise NotImplementedError('invalid model: {}'.format(model_name))

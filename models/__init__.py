from configs import CFG
from .resnet import ResNet


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
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))

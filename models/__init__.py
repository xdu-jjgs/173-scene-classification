from configs import CFG
from .unet import UNet
from .danet import DANet
from .segnet import SegNet
from .pspnet import PSPNet
from .brrnet import BRRNet
from .farseg import FarSeg
from .deeplabv3plus import DeeplabV3Plus
from .deeplabv3 import DeepLabV3ResNet18, DeepLabV3ResNet34, DeepLabV3ResNet50, DeepLabV3ResNet101


def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'deeplabv3+resnet18':
        return DeepLabV3ResNet18(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet34':
        return DeepLabV3ResNet34(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet50':
        return DeepLabV3ResNet50(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet101':
        return DeepLabV3ResNet101(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'unet':
        return UNet(num_channels, num_classes, with_bn=False)
    elif CFG.MODEL.NAME == 'unet+bn':
        return UNet(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'brrnet':
        return BRRNet(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3plus+xception':
        return DeeplabV3Plus(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'segnet':
        return SegNet(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'pspnet':
        return PSPNet(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'danet':
        return DANet(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'farseg':
        return FarSeg(num_channels, num_classes)

    else:
        raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))

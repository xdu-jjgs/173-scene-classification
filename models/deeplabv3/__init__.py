from .deeplabv3 import DeepLabV3ResNet18, DeepLabV3ResNet34, DeepLabV3ResNet50, DeepLabV3ResNet101

__all__ = [
    "DeepLabV3ResNet18", "DeepLabV3ResNet34", "DeepLabV3ResNet50", "DeepLabV3ResNet101"
]

'''
python train.py configs/deeplabv3+resnet50_sigmoid+dice_adam_plateau_40.yaml --path runs/deeplabv3-train --device cuda:0
python test.py runs/deeplabv3-train/config.yaml runs/deeplabv3-train/best.pth --path runs/deeplabv3-test-best-tmp --device cuda:1
'''

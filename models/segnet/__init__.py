from .segnet import SegNet

__all__ = [
    'SegNet'
]

'''
python train.py configs/segnet_sigmoid+dice_sgd_40.yaml --path runs/segnet-train --device cuda:0
python test.py runs/segnet-train/config.yaml runs/segnet-train/best.pth --path runs/segnet-test-best --device cuda:0
python test.py runs/segnet-train/config.yaml runs/segnet-train/last.pth --path runs/segnet-test-last --device cuda:1

python inference.py runs/segnet-train/config.yaml runs/segnet-train/best.pth D:\zts\dataset\massachusetts-buildings-dataset\png\test\22828930_15.png --output ./segnet-output.tif --device cuda:0 --no-show
'''

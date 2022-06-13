from .deeplabv3plus import DeeplabV3Plus

__all__ = [
    'DeeplabV3Plus'
]

'''
python train.py configs/deeplabv3plus+xception_sigmoid+dice_adam_plateau_40.yaml --path runs/deeplabv3plus+xception-train --device cuda:0 

python inference.py runs/deeplabv3plus+xception-train/config.yaml runs/deeplabv3plus+xception-train/best.pth D:\zts\dataset\massachusetts-buildings-dataset\png\test\22828930_15.png --output ./deeplabv3plus+xception-output.tif --device cuda:0 --no-show
'''

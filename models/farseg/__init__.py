from .farseg import FarSeg

__all__ = [
    'FarSeg'
]

'''
python train.py configs/farseg_sigmoid+dice_adam_plateau_40.yaml --path runs/farseg-train --device cuda:0
python test.py runs/farseg-train/config.yaml runs/farseg-train/best.pth --path runs/farseg-test-best --device cuda:0
python test.py runs/farseg-train/config.yaml runs/farseg-train/last.pth --path runs/farseg-test-last --device cuda:1

python train.py configs/farseg_sigmoid+dice_adam_plateau_40.yaml --path runs/farseg-train --device cuda:0
'''

from .danet import DANet

__all__ = [
    'DANet'
]

'''
python train.py configs/danet_sigmoid+dice_adam_plateau_40.yaml --path runs/danet-train --device cuda:0
python test.py runs/danet-train/config.yaml runs/danet-train/best.pth --path runs/danet-test-best --device cuda:0
python test.py runs/danet-train/config.yaml runs/danet-train/last.pth --path runs/danet-test-last --device cuda:1
'''

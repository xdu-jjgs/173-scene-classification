from .brrnet import BRRNet

__all__ = [
    "BRRNet"
]


'''
python test.py runs/brrnet-train/config.yaml runs/brrnet-train/best.pth --path runs/brrnet-test-best-tmp --device cuda:1
'''
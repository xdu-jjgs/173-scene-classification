from .unet import UNet

__all__ = [
    'UNet'
]

'''
python test.py runs/unet-train/config.yaml runs/unet-train/best.pth --path runs/unet-test-best-tmp --device cuda:1
python train.py configs/gf2-building/unet_ce_adam_plateau_40.yaml ^
                  --path ./runs/gf2-building/unet-train ^
                  --no-validate ^
                  --nodes 1 ^
                  --gpus 2 ^
                  --rank-node 0 ^
                  --backend gloo ^
                  --master-ip localhost ^
                  --master-port 8888 ^
                  --seed 30 ^
                  --opt-level O0
'''

from .pspnet import PSPNet

__all__ = [
    'PSPNet'
]

'''
python train.py configs/pspnet_sigmoid+dice_sgd_40.yaml --path runs/pspnet-train-tmp --device cuda:0 
python test.py runs/pspnet-train/config.yaml runs/pspnet-train/best.pth --path runs/pspnet-test-best --device cuda:0
python test.py runs/pspnet-train/config.yaml runs/pspnet-train/last.pth --path runs/pspnet-test-last --device cuda:1
python inference.py runs/pspnet-train/config.yaml runs/pspnet-train/best.pth D:\zts\dataset\massachusetts-buildings-dataset\png\test\22828930_15.png --output ./pspnet-output.tif --device cuda:0 --no-show
python train.py configs/gf2-building/pspnet_sigmoid+dice_adam_plateau_40.yaml ^
                  --path ./runs/gf2-building/pspnet-train ^
                  --no-validate ^
                  --nodes 1 ^
                  --gpus 1 ^
                  --rank-node 0 ^
                  --backend gloo ^
                  --master-ip localhost ^
                  --master-port 8888 ^
                  --seed 30 ^
                  --opt-level O0
'''

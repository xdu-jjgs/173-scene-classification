from .resnet import ResNet

__all__ = [
    'ResNet'
]

'''
python train.py configs/sar_msi/resnet18_3090.yaml ^
        --path ./runs/sar_msi/resnet18-train ^
        --nodes 1 ^
        --gpus 2 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi/resnet18-train/config.yaml ^
        runs/sar_msi/resnet18-train/best.pth ^
        --path runs/sar_msi/resnet18-test ^
        --device cuda:0
'''

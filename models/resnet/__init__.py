from .resnet import ResNet

__all__ = [
    'ResNet'
]

'''
python train.py configs/sar_msi/resnet18_3090.yaml ^
--path ./runs/sar_msi/resnet18-train ^
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

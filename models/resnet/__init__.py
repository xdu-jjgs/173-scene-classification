from .resnet import ResNet

__all__ = [
    ResNet
]

'''
python train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce.yaml ^
        --path ./runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/last.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-test-last ^
        --device cuda:0
        
python train.py configs/vnr_msi/resnet101_ce.yaml ^
        --path ./runs/vnr_msi/resnet101_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0 

python test.py runs/vnr_msi/resnet101_ce-train/config.yaml ^
        runs/vnr_msi/resnet101_ce-train/last.pth ^
        --path runs/vnr_msi/resnet34_ce-test-last ^
        --device cuda:0
'''

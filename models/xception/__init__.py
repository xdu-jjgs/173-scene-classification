from .xception import Xception

__all__ = [
    Xception
]

'''
python train.py configs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce.yaml ^
        --path ./runs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce-train/config.yaml ^
        runs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce-train/best.pth ^
        --path runs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce-test-best ^
        --device cuda:0

python train.py configs/vnr_msi/xception_ce.yaml ^
        --path ./runs/vnr_msi/xception_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0 

python test.py runs/vnr_msi/xception_ce-train/config.yaml ^
        runs/vnr_msi/xception_ce-train/last.pth ^
        --path runs/vnr_msi/xception_ce-test-last ^
        --device cuda:0
'''

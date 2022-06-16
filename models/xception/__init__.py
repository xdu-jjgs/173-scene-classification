from .xception import Xception

__all__ = [
    Xception
]

'''
python train.py configs/sar_msi/xception.yaml ^
        --path ./runs/sar_msi/xception-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi/xception-train/config.yaml ^
        runs/sar_msi/xception-train/last.pth ^
        --path runs/sar_msi/xception-test-last ^
        --device cuda:0

python train.py configs/vnr_msi/xception.yaml ^
        --path ./runs/vnr_msi/xception-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0 

python test.py runs/vnr_msi/xception-train/config.yaml ^
        runs/vnr_msi/xception-train/best.pth ^
        --path runs/vnr_msi/xception-test-best ^
        --device cuda:0
'''

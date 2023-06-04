from .omoe import OMOE

__all__ = [
    OMOE,
]

'''
python train/classifier/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/omoe.yaml ^
        --path ./runs/sar_msi_5000_average_0,6,8,10,15,16/omoe-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/omoe-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/omoe-train/last.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/omoe-test-last ^
        --device cuda:0
        
python train/classifier/train.py configs/vnr_msi/omoe.yaml ^
        --path ./runs/vnr_msi/omoe-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0 

python test.py runs/vnr_msi/omoe-train/config.yaml ^
        runs/vnr_msi/omoe-train/last.pth ^
        --path runs/vnr_msi/omoe-test-last ^
        --device cuda:0
'''

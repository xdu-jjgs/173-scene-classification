from .omoe import OMOEFusion

__all__ = [
    OMOEFusion,
]

'''
python train/omoe_fusion/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/omoe_fusion_18+34.yaml ^
        --path ./runs/sar_msi_5000_average_0,6,8,10,15,16/omoe_fusion_18+34-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/omoe_fusion_18+34-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/omoe_fusion_18+34-train/best.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/omoe_fusion_18+34-test-best ^
        --device cuda:0
        
python train/omoe/train.py configs/vnr_msi/omoe.yaml ^
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
        runs/vnr_msi/omoe-train/best.pth ^
        --path runs/vnr_msi/omoe-test-best ^
        --device cuda:0
'''

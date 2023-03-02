cd E:\zts\173-scene-classification
set PYTHONPATH=%cd%

rem ResNet
python train/classifier/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet18_ce.yaml ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/resnet18_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python train/fusenet/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_fusenet_ce.yaml ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_fusenet_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python test_hh.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce_tmp.yaml ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/adaboost/resnet34_ce-test-best_hh ^
        --device cuda:0

rem Xception
python train/classifier/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce.yaml ^
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

rem DenseNet
python train/classifier/train.py configs/sar_msi_5000_average_0,6,8,10,15,16/densenet121_ce.yaml ^
        --path ./runs/sar_msi_5000_average_0,6,8,10,15,16/densenet121_ce-train ^
        --nodes 1 ^
        --gpus 2 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/densenet121_ce-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/densenet121_ce-train/last.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/densenet121_ce-test-last ^
        --device cuda:0

rem ResNet
python train/classifier/train.py configs/sar_msi_15000_average_all/resnet34_ce.yaml ^
        --path runs/sar_msi_15000_average_all/resnet34_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python test.py runs/sar_msi_15000_average_all/resnet34_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet34_ce-train/last.pth ^
        --path runs/sar_msi_15000_average_all/resnet34_ce-test-last ^
        --device cuda:0

python test.py runs/sar_msi_15000_average_all/resnet34_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet34_ce-train/best.pth ^
        --path runs/sar_msi_15000_average_all/resnet34_ce-test-best ^
        --device cuda:0
call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\173-scene-classification
set PYTHONPATH=%cd%

rem ResNet34
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

rem ResNet50
python train/classifier/train.py configs/sar_msi_15000_average_all/resnet50_ce.yaml ^
        --path runs/sar_msi_15000_average_all/resnet50_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python test.py runs/sar_msi_15000_average_all/resnet50_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet50_ce-train/last.pth ^
        --path runs/sar_msi_15000_average_all/resnet50_ce-test-last ^
        --device cuda:0

python test.py runs/sar_msi_15000_average_all/resnet50_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet50_ce-train/best.pth ^
        --path runs/sar_msi_15000_average_all/resnet50_ce-test-best ^
        --device cuda:0

rem ResNet101
python train/classifier/train.py configs/sar_msi_15000_average_all/resnet101_ce.yaml ^
        --path runs/sar_msi_15000_average_all/resnet101_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1

python test.py runs/sar_msi_15000_average_all/resnet101_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet101_ce-train/last.pth ^
        --path runs/sar_msi_15000_average_all/resnet101_ce-test-last ^
        --device cuda:0

python test.py runs/sar_msi_15000_average_all/resnet101_ce-train/config.yaml ^
        runs/sar_msi_15000_average_all/resnet101_ce-train/best.pth ^
        --path runs/sar_msi_15000_average_all/resnet101_ce-test-best ^
        --device cuda:0
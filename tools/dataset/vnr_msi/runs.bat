cd E:\zts\173-scene-classification
rem ResNet18
python train.py configs/vnr_msi/resnet101_ce.yaml --path ./runs/vnr_msi/resnet18_ce-train ^
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

rem ResNet101
python train.py configs/vnr_msi/resnet101_ce.yaml --path ./runs/vnr_msi/resnet18_ce-train ^
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

rem Xception
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
call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\173-scene-classification

rem ResNet18
python train.py configs/vnr_msi_extend/resnet18_ce.yaml --path ./runs/vnr_msi/resnet18_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/vnr_msi_extend/resnet18_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet18_ce-train/last.pth ^
        --path runs/vnr_msi_extend/resnet18_ce-test-last ^
        --device cuda:0

python test.py runs/vnr_msi_extend/resnet18_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet18_ce-train/best.pth ^
        --path runs/vnr_msi_extend/resnet18_ce-test-best ^
        --device cuda:0

rem ResNet34
python train.py configs/vnr_msi_extend/resnet34_ce.yaml --path ./runs/vnr_msi/resnet34_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/vnr_msi_extend/resnet34_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet34_ce-train/last.pth ^
        --path runs/vnr_msi_extend/resnet34_ce-test-last ^
        --device cuda:0

python test.py runs/vnr_msi_extend/resnet34_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet34_ce-train/best.pth ^
        --path runs/vnr_msi_extend/resnet34_ce-test-best ^
        --device cuda:0

rem ResNet50
python train.py configs/vnr_msi_extend/resnet50_ce.yaml --path ./runs/vnr_msi/resnet50_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/vnr_msi_extend/resnet50_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet50_ce-train/last.pth ^
        --path runs/vnr_msi_extend/resnet50_ce-test-last ^
        --device cuda:0

python test.py runs/vnr_msi_extend/resnet50_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet50_ce-train/best.pth ^
        --path runs/vnr_msi_extend/resnet50_ce-test-best ^
        --device cuda:0

rem ResNet101
python train.py configs/vnr_msi_extend/resnet101_ce.yaml --path ./runs/vnr_msi_extend/resnet101_ce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0

python test.py runs/vnr_msi_extend/resnet101_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet101_ce-train/last.pth ^
        --path runs/vnr_msi_extend/resnet101_ce-test-last ^
        --device cuda:0

python test.py runs/vnr_msi_extend/resnet101_ce-train/config.yaml ^
        runs/vnr_msi_extend/resnet101_ce-train/best.pth ^
        --path runs/vnr_msi_extend/resnet101_ce-test-best ^
        --device cuda:0


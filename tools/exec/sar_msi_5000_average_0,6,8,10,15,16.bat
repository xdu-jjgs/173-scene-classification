call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\173-scene-classification
set PYTHONPATH=%cd%

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/last.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-test-last ^
        --device cuda:0

python test.py runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/config.yaml ^
        runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-train/best.pth ^
        --path runs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce-test-best ^
        --device cuda:0
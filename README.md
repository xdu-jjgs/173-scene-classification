# 173-scene-classification
173场景分类

## <a name='requirements'> </a>配置
- h5py
- numpy
- torch 
- scipy 
- sklearn 
- scikit-learn

## <a name='task'> </a>任务描述
1. 对数据集SAR-MSI和VNR-MSI进行数据融合
2. 对融合后的数据进行场景分类

## <a name='dataset'> </a>数据集描述
###<a name='dataset-sm'> </a>SAR-MSI
数据集详情：https://github.com/zhu-xlab/So2Sat-LCZ42

####类别：
1. 紧密型高层建筑
2. 紧密型中层建筑
3. 紧密型低层建筑
4. 稀疏型高层建筑
5. 稀疏型中层建筑
6. 稀疏型低层建筑
7. 轻型低层建筑
8. 大型低层建筑
9. 稀疏建筑
10. 大型工厂
11. 密集树木
12. 点型树木
13. 灌木丛
14. 低矮植物
15. 石头地
16. 沙漠地
17. 水域
####数据量：
训练集：352366，验证集：24119，测试集：24188

sen1:32 * 32 * 8

sen2:32 * 32 * 10

label:N * 17, one-hot

###<a name='dataset-vm'> </a> VNR-MSI
TODO

## <a name='structure'> </a>项目结构
TODO

## <a name='preprocess'> </a>数据预处理
### SAR_MSI数据集
包括：
1. 筛选数据
2. 重编号标签
3. 转为Tensor
4. Z-Score归一化
```shell
python preprocess_by_dataset.py configs/preprocess/sar_msi_2000.yaml ^
      --path E:/zts/dataset/SAR_MSI_preprocessed
```
### VNR_MSI数据集

包括：

1. 转为Tensor
2. Z-Score归一化

```shell
python preprocess_by_class.py configs/preprocess/vnr_msi.yaml ^
        --path E:/zts/dataset/VNR_MSI_preprocessed ^
        --train-val-test-portion 0.6 0.1 0.3 ^
        --class-list building cross factory farmland highway lake river
```

## <a name='train'> </a>模型训练
```shell
python train.py configs/sar_msi/resnet18.yaml ^
        --path ./runs/sar_msi/resnet18-train ^
        --nodes 1 ^
        --gpus 2 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O0
```

## <a name='test'> </a>模型测试
```shell
python test.py runs/sar_msi/resnet18-train/config.yaml ^
        runs/sar_msi/resnet18-train/best.pth ^
        --path runs/sar_msi/resnet18-test ^
        --device cuda:0
```

## <a name='result'> </a>结果

| Dataset | Model                                     | OA-best| OA-last|
|---------|-------------------------------------------|-------|-------|
| SAR_MSI | [ResNet18](configs/sar_msi/resnet18.yaml) | 0.915 |0.898|
| SAR_MSI | [ResNet34](configs/sar_msi/resnet34.yaml) | 0.898 |0.850|
| SAR_MSI | [ResNet50](configs/sar_msi/resnet50.yaml) | 0.897 |0.898|
| SAR_MSI | [ResNet101](configs/sar_msi/resnet101.yaml) | 0.910 |0.892|
| VNR_MSI | [ResNet18](configs/vnr_msi/resnet18.yaml) | 0.745 |0.783|
| VNR_MSI | [ResNet34](configs/vnr_msi/resnet34.yaml) | 0.868 |0.877|
| VNR_MSI | [ResNet50](configs/vnr_msi/resnet50.yaml) | 0.708 |0.811|
| VNR_MSI | [ResNet101](configs/vnr_msi/resnet101.yaml) | 0.708 |0.745|

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

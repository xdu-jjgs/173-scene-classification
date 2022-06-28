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

### <a name='dataset-sm'> </a>SAR-MSI

数据集详情：https://github.com/zhu-xlab/So2Sat-LCZ42

#### 类别：

0. 紧密型高层建筑
1. 紧密型中层建筑
2. 紧密型低层建筑
3. 稀疏型高层建筑
4. 稀疏型中层建筑
5. 稀疏型低层建筑
6. 轻型低层建筑
7. 大型低层建筑
8. 稀疏建筑
9. 大型工厂
10. 密集树木
11. 点型树木
12. 灌木丛
13. 低矮植物
14. 石头地
15. 沙漠地
16. 水域

#### 数据量：

训练集：352366，验证集：24119，测试集：24188

sen1:32 * 32 * 8

sen2:32 * 32 * 10

label:N * 17, one-hot

### <a name='dataset-vm'> </a> VNR-MSI

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
python preprocess_by_dataset.py configs/preprocess/sar_msi_2000_average_0,9,10,13,14,16.yaml ^
      --path E:/zts/dataset/SAR_MSI_preprocessed_2000_average
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
python train.py configs/sar_msi_2000_average_0,9,10,13,14,16/resnet18_ce.yaml ^
        --path ./runs/sar_msi_2000_average_0,9,10,13,14,16/resnet18_ce-train ^
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
python test.py runs/sar_msi_2000_average_0,9,10,13,14,16/resnet18_ce-train/config.yaml ^
        runs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet18_ce-train/best.pth ^
        --path runs/sar_msi_2000_average_0,9,10,13,14,16/resnet18-test ^
        --device cuda:0
```

## <a name='result'> </a>结果

| Dataset               | Model                                                                           | loss       | OA-best | AA-best | OA-last | AA-last |
|-----------------------|---------------------------------------------------------------------------------|------------|---------|---------|---------|---------|
| SAR_MSI_2000_sequence | [ResNet18](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet18_ce.yaml)      | softmax+ce | 0.915   | -       | 0.898   | -       |
| SAR_MSI_2000_sequence | [ResNet34](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet34_ce.yaml)      | softmax+ce | 0.898   | -       | 0.850   | -       |
| SAR_MSI_2000_sequence | [ResNet34](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet34_focal.yaml)   | softmax+ce | 0.897   | -       | 0.890   | -       |
| SAR_MSI_2000_sequence | [ResNet50](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet50_ce.yaml)      | softmax+ce | 0.897   | 0.708   | 0.898   | 0.851   |
| SAR_MSI_2000_sequence | [ResNet50](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet50_focal.yaml)   | focal      | 0.892   | -       | 0.897   | -       |   
| SAR_MSI_2000_sequence | [ResNet101](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet101_ce.yaml)    | softmax+ce | 0.910   | 0.804   | 0.892   | 0.782   |
| SAR_MSI_2000_sequence | [ResNet101](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet101_focal.yaml) | focal      | 0.898   | -       | 0.883   | -       | 
| SAR_MSI_2000_sequence | [Xception](configs/sar_msi_2000_sequence_0,9,10,13,14,16/xception_ce.yaml)      | softmax+ce | 0.905   | -       | 0.903   | 0.810   |
| SAR_MSI_2000_average  | [ResNet18](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet18_ce.yaml)       | softmax+ce | 0.803   | 0.812   | 0.778   | 0.798   |
| SAR_MSI_2000_average  | [ResNet34](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet34_ce.yaml)       | softmax+ce | 0.760   | 0.777   | 0.762   | 0.777   |
| SAR_MSI_2000_average  | [ResNet50](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet50_ce.yaml)       | softmax+ce | 0.762   | 0.776   | 0.700   | 0.716   |
| SAR_MSI_2000_average  | [ResNet101](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce.yaml)     | softmax+ce | 0.755   | 0.774   | 0.763   | 0.777   |
| VNR_MSI               | [ResNet18](configs/vnr_msi/resnet18_ce.yaml)                                    | softmax+ce | 0.745   | 0.      | 0.783   | 0.      |
| VNR_MSI               | [ResNet34](configs/vnr_msi/resnet34_ce.yaml)                                    | softmax+ce | 0.868   | 0.      | 0.877   | 0.      |
| VNR_MSI               | [ResNet50](configs/vnr_msi/resnet50_ce.yaml)                                    | softmax+ce | 0.708   | 0.      | 0.811   | 0.      |
| VNR_MSI               | [ResNet101](configs/vnr_msi/resnet101_ce.yaml)                                  | softmax+ce | 0.708   | 0.      | 0.745   | 0.      |
| VNR_MSI               | [Xception](configs/vnr_msi/xception_ce.yaml)                                    | softmax+ce | 0.792   | 0.      | 0.858   | 0.      |

## <a name="todo"></a> ToDO

- [x] 数据选择：类别平衡
- [ ] 数据增强
- [x] 损失函数：Focal Loss
- [ ] 更换SAR_MSI选择类别 保留0-compact high-rise, 10-dense trees, 16-water 排除9-heavy industry, 13-bush/scrub, 14-bare
  rock/paved

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

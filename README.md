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

#### <a name='dataset-sm-class'> </a>类别

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

#### <a name='dataset-sm-amount'> </a>数据量

训练集：352366，验证集：24119，测试集：24188

sen1:32 * 32 * 8

sen2:32 * 32 * 10

label:N * 17, one-hot

#### <a name='dataset-sm-subdataset'> </a>子数据集

1. sub1:

   sequence

   train-val-test:1000-400-600

   0, 9, 10, 13, 14, 16

   compact high-rise、heavy industry、dense trees、low plants、bare rock/ paved、water

2. sub2:

   average

   train-val-test:1000-400-600

   0, 9, 10, 13, 14, 16

   compact high-rise、heavy industry、dense trees、low plants、bare rock/ paved、water

3. sub3:

   average

   train-val-test:2000-400-600

   0, 6, 8, 10, 15, 16

   compact high-rise、lightweight low-rise、sparsely built、dense trees、bare soil/ sand、water

4. sub4:

   average

   train-val-test:3000-1000-1000

   0, 6, 7, 10, 15, 16

   compact high-rise、lightweight low-rise、large low-rise、dense trees、bare soil/ sand、water

5. sub5:
   average

   train-val-test:3000-1000-1000

   0, 6, 8, 10, 15, 16

   compact high-rise、lightweight low-rise、sparsely built、dense trees、bare soil/ sand、water

### <a name='dataset-vm'> </a> VNR-MSI

### <a name='dataset-vm-gf2'> </a> GF-2

高分二号多光谱图像数据由我国自主研制的高分二号（GF-2）卫星采集得到， 包含波长范围为0.45~0.52um、0.52~0.59um、0.63~0.69um、0.77~0.89um的四个波段图像信息，
空间分辨率为4m，成像区域为山东省聊城市中心市区及周边区县， 经纬度范围为36°11'3.83"N,115°38'12.33"E至36°34'55.23"N,116°17'54.16"E， 图像覆盖地物类别主要为水体、建筑、耕地、道路。

### <a name='dataset-vm-terra'> </a> Terra

Terra红外图像数据由Terra卫星搭载的ASTER传感器采集得到， 包含波长范围为0.52~0.60um、0.63~0.69um、0.78~0.86um的可见光近红外波段共三个，
空间分辨率为15m，成像区域为山东聊城市中心市区及周边区县， 经纬度范围为36°10'42.87"N,115°42'55.29"E至36°38'5.77"N,116°33'26.24"E， 图像覆盖和高分二号多光谱图像数据基本一致。

#### <a name='dataset-sm-class'> </a>类别

1. 建筑
2. 路口
3. 工厂
4. 耕地
5. 公路
6. 湖泊
7. 河流

#### <a name='dataset-vm-amount'> </a>数据量

共337个样本,各类别分别为52+51+53+53+52+24+52

gf2:256 * 256 * 4

terra:256 * 256 * 3

## <a name='preprocess'> </a>数据预处理

### <a name='preprocess-sm'> </a>SAR_MSI数据集

包括：

1. 筛选数据
2. 重编号标签
3. 转为Tensor
4. Z-Score归一化

```shell
python preprocess_by_dataset.py configs/preprocess/sar_msi_5000_average_0,6,8,10,15,16.yaml ^
      --path E:/zts/dataset/SAR_MSI_preprocessed_5000_average_0,6,8,10,15,16
```

### <a name='preprocess-vm'> </a>VNR_MSI数据集

#### 数据集制作

```shell
python tools/dataset/vnr_msi/assemble.py ^
        D:/dataset/VNR_Raw/output_MATLAB ^
        --path D:/dataset/VNR_Raw/output_assemble ^ 
        --class-list building cross factory farmland highway lake river
```

#### 预处理

包括：

1. 转为Tensor
2. Z-Score归一化

```shell
python preprocess_by_class.py configs/preprocess/vnr_msi_extend.yaml ^
        --path E:/zts/dataset/VNR_MSI_extend_preprocessed ^
        --train-val-test-portion 0.7 0.1 0.2 ^
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

| Dataset      | Model                                                                           | loss       | OA-best | AA-best | worst-best | OA-last | AA-last | worst-last |
|--------------|---------------------------------------------------------------------------------|------------|---------|---------|------------|---------|---------|------------|
| SAR_MSI_sub1 | [ResNet18](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet18_ce.yaml)      | softmax+ce | 0.915   | -       | -          | 0.898   | -       | -          |
| SAR_MSI_sub1 | [ResNet34](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet34_ce.yaml)      | softmax+ce | 0.898   | -       | -          | 0.850   | -       | -          |
| SAR_MSI_sub1 | [ResNet34](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet34_focal.yaml)   | softmax+ce | 0.897   | -       | -          | 0.890   | -       | -          |
| SAR_MSI_sub1 | [ResNet50](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet50_ce.yaml)      | softmax+ce | 0.897   | 0.708   | -          | 0.898   | 0.851   | 0.614      |
| SAR_MSI_sub1 | [ResNet50](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet50_focal.yaml)   | focal      | 0.892   | -       | -          | 0.897   | -       | -          |   
| SAR_MSI_sub1 | [ResNet101](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet101_ce.yaml)    | softmax+ce | 0.910   | 0.804   | 0.421      | 0.892   | 0.782   | 0.471      |
| SAR_MSI_sub1 | [ResNet101](configs/sar_msi_2000_sequence_0,9,10,13,14,16/resnet101_focal.yaml) | focal      | 0.898   | -       | -          | 0.883   | -       | -          | 
| SAR_MSI_sub1 | [Xception](configs/sar_msi_2000_sequence_0,9,10,13,14,16/xception_ce.yaml)      | softmax+ce | 0.905   | -       | -          | 0.903   | 0.810   | 0.         |
| SAR_MSI_sub2 | [ResNet18](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet18_ce.yaml)       | softmax+ce | 0.803   | 0.812   | 0.558      | 0.778   | 0.798   | 0.534      |
| SAR_MSI_sub2 | [ResNet34](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet34_ce.yaml)       | softmax+ce | 0.760   | 0.777   | 0.520      | 0.762   | 0.777   | 0.530      |
| SAR_MSI_sub2 | [ResNet50](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet50_ce.yaml)       | softmax+ce | 0.762   | 0.776   | 0.545      | 0.700   | 0.716   | 0.487      |
| SAR_MSI_sub2 | [ResNet101](configs/sar_msi_2000_average_0,9,10,13,14,16/resnet101_ce.yaml)     | softmax+ce | 0.755   | 0.774   | 0.471      | 0.763   | 0.777   | 0.485      |
| SAR_MSI_sub3 | [ResNet18](configs/sar_msi_3000_average_0,6,8,10,15,16/resnet18_ce.yaml)        | softmax+ce | 0.902   | 0.907   | 0.784      | 0.895   | 0.899   | 0.773      |
| SAR_MSI_sub3 | [ResNet34](configs/sar_msi_3000_average_0,6,8,10,15,16/resnet34_ce.yaml)        | softmax+ce | 0.920   | 0.923   | 0.841      | 0.887   | 0.896   | 0.736      |
| SAR_MSI_sub3 | [ResNet50](configs/sar_msi_3000_average_0,6,8,10,15,16/resnet50_ce.yaml)        | softmax+ce | 0.837   | 0.849   | 0.688      | 0.783   | 0.837   | 0.531      |
| SAR_MSI_sub3 | [ResNet101](configs/sar_msi_3000_average_0,6,8,10,15,16/resnet101_ce.yaml)      | softmax+ce | 0.865   | 0.867   | 0.792      | 0.833   | 0.843   | 0.712      |
| SAR_MSI_sub4 | [ResNet18](configs/sar_msi_3000_average_0,6,7,10,15,16/resnet18_ce.yaml)        | softmax+ce | 0.882   | 0.886   | 0.780      | 0.882   | 0.886   | 0.780      |
| SAR_MSI_sub4 | [ResNet34](configs/sar_msi_3000_average_0,6,7,10,15,16/resnet34_ce.yaml)        | softmax+ce | 0.898   | 0.902   | 0.800      | 0.863   | 0.879   | 0.686      |
| SAR_MSI_sub4 | [ResNet50](configs/sar_msi_3000_average_0,6,7,10,15,16/resnet50_ce.yaml)        | softmax+ce | 0.847   | 0.852   | 0.702      | 0.852   | 0.858   | 0.724      |
| SAR_MSI_sub5 | [ResNet18](configs/sar_msi_5000_average_0,6,8,10,15,16/resnet18_ce.yaml)        | softmax+ce | 0.905   | 0.907   | 0.841      | 0.911   | 0.912   | 0.815      |
| SAR_MSI_sub5 | [ResNet34](configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce.yaml)        | softmax+ce | 0.906   | 0.908   | 0.860      | 0.881   | 0.883   | 0.765      |
| VNR_MSI      | [ResNet18](configs/vnr_msi/resnet18_ce.yaml)                                    | softmax+ce | 0.745   | 0.      | 0.783      | 0.      |
| VNR_MSI      | [ResNet34](configs/vnr_msi/resnet34_ce.yaml)                                    | softmax+ce | 0.868   | 0.      | 0.877      | 0.      |
| VNR_MSI      | [ResNet50](configs/vnr_msi/resnet50_ce.yaml)                                    | softmax+ce | 0.708   | 0.      | 0.811      | 0.      |
| VNR_MSI      | [ResNet101](configs/vnr_msi/resnet101_ce.yaml)                                  | softmax+ce | 0.708   | 0.      | 0.745      | 0.      |
| VNR_MSI      | [Xception](configs/vnr_msi/xception_ce.yaml)                                    | softmax+ce | 0.792   | 0.      | 0.858      | 0.      |

## <a name="todo"></a> ToDO

- [x] 数据选择：类别平衡
- [ ] 数据增强
- [x] 损失函数：Focal Loss
- [x] 更换SAR_MSI选择类别 保留0-compact high-rise, 10-dense trees, 16-water 排除9-heavy industry, 13-bush/scrub, 14-bare
  rock/paved

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

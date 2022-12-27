import os
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime

from configs import CFG
from metric import Metric
from models import build_model
from datas import build_dataset, build_dataloader
import pickle
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for test')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    args = parser.parse_args()
    return args


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main(data_file_path, checkpoint_path: str):
    # parse command line arguments

    args = parse_args()
    # 确保checkpoint存在

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'test.log')),
            logging.StreamHandler(),
        ])

    # build dataset
    class_interest = CFG.DATASET.CLASSES_INTEREST
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = len(class_interest)
    # build data loader
    # test_dataloader = build_dataloader(test_dataset, 'test')
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(args.device)
    # build metric
    metric = Metric(NUM_CLASSES)

    # load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError('checkpoint {} not found'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # delete module saved in train
    model.load_state_dict(({k.replace('module.', ''): v for k, v in checkpoint['model']['state_dict'].items()}))
    best_PA = checkpoint['metric']['PA']
    logging.info('load checkpoint {} with PA={:.3f}'.format(checkpoint, best_PA))

    # test
    model.eval()  # set model to evaluation mode
    metric.reset()  # reset metric
    # test_bar = tqdm(test_dataloader, desc='testing', ascii=True)
    file_list = os.listdir(data_file_path)

    name = []
    err_rate = []
    with torch.no_grad():  # disable gradient back-propagation
        for filename in tqdm(file_list):
            sample_path = os.path.join(data_file_path, filename)
            fo = open(sample_path, 'rb')
            data = pickle.load(fo)
            sen1 = data['data1']
            sen2 = data['data2']
            label = data['label']
            x = np.concatenate((sen1, sen2), axis=0)
            x, label = torch.tensor(x), torch.tensor(label)
            # print(x.shape())
            x = x.unsqueeze(0)
            # sens = np.array(sens)
            x, label = x.float().to(args.device), label.to(args.device)
            y = model(x)
            y = y.squeeze()

            pred = y.data.cpu().numpy()
            # print(pred)
            pred = softmax(pred)
            gt = np.array(label.cpu())
            err = np.sum(pred) - pred[gt] + (1 - pred[gt])

            name.append(filename)
            err_rate.append(err)
    return name, err_rate


def generate_base(data_file_path, checkpoint_path: str = None):
    filename, err = main(data_file_path, checkpoint_path)
    err_rate = err / np.max(err)
    if np.mean(err_rate) > 0.5:
        err_rate -= 2 * (np.mean(err_rate) - 0.5)
    mean = np.mean(err_rate)
    p = np.zeros((len(filename), 1))
    p = p.astype(float)
    p += 1 / len(filename)
    # print(np.shape(err_rate))
    Err_rate = np.sum(np.dot(p.T, err_rate))
    weight1 = np.log((1 - Err_rate) / Err_rate)
    # print('weight:' + str(weight1))

    a = Err_rate / (1 - Err_rate)
    # print(a)
    # # print(name.shape[0])
    j = 0
    for i in range(len(filename)):
        if err_rate[i] < mean:
            j += 1
            p[i, 0] = p[i, 0] * a
    p = p / np.sum(p)
    p = p.squeeze()
    sample = np.random.choice(filename, len(filename), replace=True, p=p)
    return weight1, sample, p


def generate_boost(data_file_path, p, checkpoint_path: str = None):
    p = p[:, None]
    filename, err = main(data_file_path, checkpoint_path)
    err_rate = err / np.max(err)
    if np.mean(err_rate) > 0.5:
        err_rate -= 2 * (np.mean(err_rate) - 0.5)
    mean = np.mean(err_rate)
    # print(np.shape(err_rate))
    Err_rate = np.sum(np.dot(p.T, err_rate))
    weight1 = np.log((1 - Err_rate) / Err_rate)
    # print('weight:' + str(weight1))

    a = Err_rate / (1 - Err_rate)
    # print(a)
    # # print(name.shape[0])
    j = 0
    for i in range(len(filename)):
        if err_rate[i] < mean:
            j += 1
            p[i, 0] = p[i, 0] * a
    p = p / np.sum(p)
    p = p.squeeze()
    sample = np.random.choice(filename, len(filename), replace=True, p=p)
    return weight1, sample, p


def generate_new_dataset(data_file_path_base, new_data_path, sample):
    os.makedirs(new_data_path, exist_ok=True)
    for file in tqdm(sample):
        src = os.path.join(data_file_path_base, file)
        filelist = os.listdir(new_data_path)
        while (file in filelist):
            file = str(np.random.randint(0, 100)) + '_' + file
        dis = os.path.join(new_data_path, file)
        shutil.copy(src, dis)


if __name__ == '__main__':
    data_file_path_base = r'E:\zts\dataset\SAR_MSI_preprocessed_2000_average\train'
    new_data_file_path_1 = r'E:\zts\dataset\SAR_MSI_preprocessed_2000_average\train_1'
    new_data_file_path_2 = r'E:\zts\dataset\SAR_MSI_preprocessed_2000_average\train_2'
    new_data_file_path_3 = r'E:\zts\dataset\SAR_MSI_preprocessed_2000_average\train_3'
    checkpoint_path = [
        'runs/sar_msi_5000_average_0,6,8,10,15,16/adaboost/resnet34_ce-train_1',
        'runs/sar_msi_5000_average_0,6,8,10,15,16/adaboost/resnet34_ce-train_2',
        'runs/sar_msi_5000_average_0,6,8,10,15,16/adaboost/resnet34_ce-train_3',
    ]
    # weight, sample, p 分别为各个模型的投票权重，新样本的抽样结果，各样本的抽样概率

    # -----都是在base的train上衡量各个模型的表现

    # -----在此可调用train_1.py，训练模型1---------------
    # --------加载模型1----------------------------

    os.system("python train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce_tmp.yaml "
              "--path {} "
              "--nodes 1 "
              "--gpus 1 "
              "--rank-node 0 "
              "--backend gloo "
              "--master-ip localhost "
              "--master-port 8888 "
              "--seed 30 "
              "--opt-level O1 ".format(checkpoint_path[0]))
    weight1, sample1, p1 = generate_base(data_file_path_base, os.path.join(checkpoint_path[0], 'best.pth'))
    print('weight1:' + str(weight1))
    print(sample1)
    generate_new_dataset(data_file_path_base, new_data_file_path_1, sample1)

    # -----在此可调用train_2.py，训练模型2---------------
    # --------加载模型2------------------------------
    os.system("python train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce_tmp.yaml "
              "--path {} "
              "--nodes 1 "
              "--gpus 1 "
              "--rank-node 0 "
              "--backend gloo "
              "--master-ip localhost "
              "--master-port 8888 "
              "--seed 30 "
              "--opt-level O1 ".format(checkpoint_path[1]))
    weight2, sample2, p2 = generate_boost(data_file_path_base, p1, os.path.join(checkpoint_path[1], 'best.pth'))
    print('weight2:' + str(weight2))
    generate_new_dataset(data_file_path_base, new_data_file_path_2, sample2)

    # -----在此可调用train_3.py，训练模型3---------------
    # --------加载模型3------------------------------
    os.system("python train.py configs/sar_msi_5000_average_0,6,8,10,15,16/resnet34_ce_tmp.yaml "
              "--path {} "
              "--nodes 1 "
              "--gpus 1 "
              "--rank-node 0 "
              "--backend gloo "
              "--master-ip localhost "
              "--master-port 8888 "
              "--seed 30 "
              "--opt-level O1 ".format(checkpoint_path[2]))
    weight3, sample3, p3 = generate_boost(data_file_path_base, p2, os.path.join(checkpoint_path[2], 'best.pth'))
    print('weight3:' + str(weight2))
    generate_new_dataset(data_file_path_base, new_data_file_path_3, sample3)

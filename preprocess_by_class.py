import os
import pickle
import argparse

import numpy as np

from configs import CFG
from datetime import datetime

from datas import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    parser.add_argument('--train-val-test-portion',
                        type=float,
                        nargs='+',
                        help='train, val, test portions')
    parser.add_argument('--class-list',
                        type=str,
                        nargs='+',
                        help='class names')
    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help='random seed')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # set random seed
    np.random.seed(args.seed)

    portion = args.train_val_test_portion
    assert abs(sum(portion) - 1.0) < 1e-5

    splits = args.class_list
    train_datasets = []
    val_datasets = []
    test_datasets = []
    # TODO: add tqdm
    for split in splits:
        dataset = build_dataset(split)
        amount = len(dataset)
        train_dataset = dataset[:int(amount * portion[0])]
        val_dataset = dataset[int(amount * portion[0]): int(amount * portion[0]) + int(amount * portion[1])]
        test_dataset = dataset[int(amount * portion[0]) + int(amount * portion[1]):]

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    splits = ['train', 'val', 'test']
    datasets = [train_datasets, val_datasets, test_datasets]
    for split, dataset in zip(splits, datasets):
        save_path = os.path.join(args.path, split)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        samples = []
        for dataset_class in dataset:
            data, label = dataset_class
            vnrs, gfs = data
            for vnr, gf in zip(vnrs, gfs):
                data = {
                    'data1': vnr,
                    'data2': gf,
                    'label': label
                }
                samples.append(data)
        np.random.shuffle(samples)
        for index, sample in enumerate(samples):
            fw = open(os.path.join(save_path, '{}_{}.pkl'.format(index, sample['label'])), 'wb')
            pickle.dump(sample, fw)
            fw.close()


if __name__ == '__main__':
    main()

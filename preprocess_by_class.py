import os
import pickle
import argparse

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
                        type=list,
                        help='class names')
    parser.add_argument('--class-list',
                        type=list,
                        help='class names')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    splits = args.class_list
    train_datasets = []
    val_datasets = []
    test_datasets = []
    # TODO: add tqdm
    for split in splits:
        save_path = os.path.join(args.path, split)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        dataset = build_dataset(split)
        amount = len(dataset)
        portion = args.train_val_test_portion
        assert abs(sum(portion) - 1.0) < 1e-5
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

        for index, sample in enumerate(dataset):
            data, label = sample
            data = {
                'sen1': data[0],
                'sen2': data[1],
                'label': label
            }
            fw = open(os.path.join(save_path, '{}_{}.pkl'.format(index, label)), 'wb')
            pickle.dump(data, fw)
            fw.close()


if __name__ == '__main__':
    main()

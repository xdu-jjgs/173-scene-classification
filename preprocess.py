import os
import argparse
import pathlib

import scipy.io as sio

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
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    splits = ['train', 'val', 'test']
    for split in splits:
        save_path = os.path.join(args.path, split)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        dataset = build_dataset(split)
        for sample in dataset:
            data, label = sample
            if data is not None:
                sio.savemat(os.path.join(save_path, '.mat'),
                            {
                                'sen1': data[0],
                                'sen2': data[1],
                                'label': label
                            })


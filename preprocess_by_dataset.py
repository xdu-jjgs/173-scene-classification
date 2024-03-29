import os
import json
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
    # TODO: add tqdm
    for index, split in enumerate(splits):
        save_path = os.path.join(args.path, split)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        sample_num = CFG.DATASET.SAMPLE_NUM[index]
        sample_order = CFG.DATASET.SAMPLE_ORDER
        assert sample_order in ['sequence', 'average']
        dataset = build_dataset(split)
        # count = 0
        count = [0] * len(CFG.DATASET.CLASSES_INTEREST)
        single_num = sample_num // len(count)
        for sample in dataset:
            data, label = sample
            if data is not None:
                if sample_order == 'average' and count[label] >= single_num:
                    continue
                # label here has been reordered
                count[label] += 1
                # print(data[0].shape, data[1].shape, label)
                data = {
                    'data1': data[0],
                    'data2': data[1],
                    'label': label
                }

                fw = open(os.path.join(save_path, '{}_{}.pkl'.format(sum(count), label)), 'wb')
                pickle.dump(data, fw)
                fw.close()

                if sample_order == 'sequence' and sum(count) >= sample_num:
                    break
                elif sample_order == 'average' and min(count) >= single_num:
                    break
        else:
            if sample_order == 'sequence':
                print("Insufficient sample number for {} dataset, expect {} but actually {}".format(split, single_num,
                                                                                                    sum(count)))
            elif sample_order == 'average':
                for ind, num in enumerate(count):
                    if num < single_num:
                        print(
                            "Insufficient sample number for class {} in {} dataset, expect {} but actually {}.".format(
                                dataset.names[ind], split, single_num, count[num]))
        print("{}: {}".format(split, count))


if __name__ == '__main__':
    main()

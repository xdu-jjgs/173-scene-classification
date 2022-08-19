import os
import argparse
import numpy as np
import scipy.io as sio

from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='input path')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    parser.add_argument('--class-list',
                        type=str,
                        nargs='+',
                        help='class names')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    files = os.listdir(args.input)
    for class_ in args.class_list:

        class_files_gf2 = list(
            filter(lambda x: x.split('_')[0] == 'gf2' and x.split('_')[-1].split('.')[0] == class_, files))
        class_data_gf2 = []
        for file in class_files_gf2:
            data = sio.loadmat(os.path.join(args.input, file))['data']
            class_data_gf2.append(data)
        class_data_gf2 = np.concatenate(class_data_gf2)
        print(class_data_gf2.shape)
        gf2_data = {
            'data': class_data_gf2
        }
        sio.savemat(os.path.join(args.path, 'gf2_{}.mat'.format(class_)), gf2_data)

        class_files_vnr = list(
            filter(lambda x: x.split('_')[0] == 'vnr' and x.split('_')[-1].split('.')[0] == class_, files))
        assert len(class_files_vnr) == 1
        data = sio.loadmat(os.path.join(args.input, class_files_vnr[0]))['data']
        vnr_data = {
            'data': data
        }
        sio.savemat(os.path.join(args.path, 'vnr_{}.mat'.format(class_)), vnr_data)


if __name__ == '__main__':
    main()

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
            filter(lambda x: 'gf2' in x and x.split('_')[-1].split('.')[0] == class_, files))
        out_gf2 = []
        out_vnr = []
        for file in class_files_gf2:
            data_gf2 = sio.loadmat(os.path.join(args.input, file))['data']
            out_gf2.append(data_gf2)

            data_vnr = sio.loadmat(os.path.join(args.input, file.replace('gf2', 'vnr')))['data']
            out_vnr.append(data_vnr)
            # if data_gf2.shape[0] != data_vnr.shape[0]:
            #     print("file {} with gf2 {}, but vnr {}".format(file, data_gf2.shape[0], data_vnr.shape[0]))
            assert data_gf2.shape[0] == data_vnr.shape[0], "file {} with gf2 {}, but vnr {}".format(file,
                                                                                                    data_gf2.shape[0],
                                                                                                    data_vnr.shape[0])
        out_gf2 = np.concatenate(out_gf2)
        out_gf2 = {
            'data': out_gf2
        }
        sio.savemat(os.path.join(args.path, '{}_{}.mat'.format('gf2', class_)), out_gf2)

        out_vnr = np.concatenate(out_vnr)
        out_vnr = {
            'data': out_vnr
        }
        sio.savemat(os.path.join(args.path, '{}_{}.mat'.format('vnr', class_)), out_vnr)

        assert out_gf2['data'].shape[0] == out_vnr['data'].shape[0], "class {}, gf2 number {} but vnr number {}".format(
            class_, out_gf2['data'].shape[0], out_vnr['data'].shape[0])
        print("class {}, number {}".format(class_, out_vnr['data'].shape[0]))


if __name__ == '__main__':
    main()

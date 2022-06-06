import h5py
import numpy as np
import scipy.io as sio

def DatasetGenerate(path, class_list):
    # read data
    vnr = np.array([], dtype=float).reshape(-1, 256, 256, 3)
    gf = np.array([], dtype=float).reshape(-1, 256, 256, 4)
    label = []
    label_index = 1
    for class_name in class_list:
        vnr_file = sio.loadmat(path + 'vnr_' + class_name + '.mat')
        gf_file = sio.loadmat(path + 'gf_' + class_name + '.mat')
        vnr_by_class = vnr_file['data']
        gf_by_class = gf_file['data']
        vnr = np.vstack([vnr, vnr_by_class])
        gf = np.vstack([gf, gf_by_class])
        num = len(vnr_by_class)
        print(num)
        label.extend([label_index] * num)
        label_index += 1
    label = np.array(label)
    return vnr, gf, label


if __name__ == '__main__':
    data_dir = "data/vnr-gf/"
    class_list = ['building', 'cross', 'factory', 'farmland', 'highway', 'lake', 'river']
    save_path = 'data/vnr-gf/dataset_vnr-fg.mat'
    # sample_num = 128  # 2^7
    # class_interest = np.array([0, 9, 10, 13, 14, 16])
    vnr, gf, label = DatasetGenerate(data_dir, class_list)
    # print('finish label transforming')
    print(vnr.shape, gf.shape, label.shape)
    # (337, 256, 256, 3) (337, 256, 256, 4) (337,)
    # sio.savemat(save_path, {'vnr': vnr, 'gf': gf, 'label': label})
    # print(label[:10])  # [13 11 10  1  1 10  7 16 15 10]
    # 最后要的label不是one-hot形式的 且label标号从1开始

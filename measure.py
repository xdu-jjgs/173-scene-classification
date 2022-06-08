import os

import h5py
import numpy as np


if __name__ == '__main__':

    root = 'E:/dataset/SAR-MSI_dataset/So2Sat_LCZ42'
    train_data = h5py.File(os.path.join(root, 'training.h5'))
    sen1 = train_data['sen1'][:5000].reshape(-1, 8)
    sen2 = train_data['sen2'][:5000].reshape(-1, 10)

    sen1_means = np.mean(sen1, axis=0)
    sen1_stds = np.std(sen1, axis=0)
    sen2_means = np.mean(sen2, axis=0)
    sen2_stds = np.std(sen2, axis=0)
    print(sen1_means, sen1_stds, sen2_means, sen2_stds)
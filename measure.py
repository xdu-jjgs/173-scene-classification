import os
import h5py
import pickle
import numpy as np
import scipy.io as sio

from collections import Counter

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

    root = r'E:/zts/dataset/SAR_MSI_preprocessed/train'
    files = [os.path.join(root, file) for file in os.listdir(root)]
    sen1s = []
    sen2s = []
    labels = []
    for file in files:
        fo = open(file, 'rb')
        data = pickle.load(fo)
        sen1 = data['sen1'].numpy()
        sen2 = data['sen2'].numpy()
        label = data['label']
        sen1s.append(sen1)
        sen2s.append(sen2)
        labels.append(label)
    sen1s = np.array(sen1s).reshape(-1, 8)
    sen2s = np.array(sen2s).reshape(-1, 10)
    print(np.mean(sen1s, axis=0), np.std(sen1s, axis=0))
    print(np.mean(sen2s, axis=0), np.std(sen2s, axis=0))
    # ({5: 313, 3: 279, 2: 276, 1: 85, 0: 28, 4: 19})
    print(Counter(labels))

    root = 'E:/zzy/173/data/vnr-gf'
    files = os.listdir(root)
    vnrs = []
    gfs = []
    for file in files:
        if file.startswith('vnr'):
            vnrs.append(sio.loadmat(file))
        elif file.startswith('gf'):
            gfs.append(sio.loadmat(file))
    vnrs = np.array(vnrs).reshape(-1, 3)
    gfs = np.array(gfs).reshape(-1, 4)



import os

import h5py
import numpy as np

from datas.base import Dataset


class RawSARMSI(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(RawSARMSI, self).__init__()
        assert split in ['train', 'val', 'test']
        if split in ['train', 'test']:
            filename = split + 'ing.h5'
        else:
            filename = 'validation.h5'
        self.data_path = os.path.join(root, filename)
        self.data = h5py.File(self.data_path, 'r')

        self.transform = transform

    def __len__(self):
        return self.data['label'].shape[0]

    def __getitem__(self, item):
        sen1 = self.data['sen1'][item]  # N*W*H*C = 352366*32*32*8
        sen2 = self.data['sen2'][item]  # N*W*H*C = 352366*32*32*10
        label = np.argmax(self.data['label'][item])  # 352366*17, one-hot

        if self.transform is not None:
            sen1, sen2, label = self.transform(sen1, sen2, label)
        sens = (sen1, sen2)
        return sens, label

import os
import h5py
import numpy as np
import scipy.io as sio

from datas.base import Dataset


class RawSARMSI(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(RawSARMSI, self).__init__()
        assert split in ['train', 'val', 'test']
        # train:352366, val:24119, test:24188
        if split in ['train', 'test']:
            filename = split + 'ing.h5'
        else:
            filename = 'validation.h5'
        self.data_path = os.path.join(root, filename)
        self.data = h5py.File(self.data_path, 'r')

        self.transform = transform

    def __getitem__(self, item):
        sen1 = self.data['sen1'][item]  # N*32*32*8
        sen2 = self.data['sen2'][item]  # N*32*32*10
        label = np.argmax(self.data['label'][item])  # N*17, one-hot
        sens = (sen1, sen2)
        if self.transform is not None:
            sens, label = self.transform(sens, label)
        return sens, label

    def __len__(self):
        return self.data['label'].shape[0]

    @property
    def num_channels(self):
        return 8, 10

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(17))

    @property
    def names(self):
        # from https://github.com/zhu-xlab/So2Sat-LCZ42
        return [
            'compact high-rise',  # 紧密型高层建筑
            'compact middle-rise',  # 紧密型中层建筑
            'compact low-rise',  # 紧密型低层建筑
            'open high-rise',  # 稀疏型高层建筑
            'open middle-rise',  # 稀疏型中层建筑
            'open low-rise',  # 稀疏型低层建筑
            'lightweight low-rise',  # 轻型低层建筑
            'large low-rise',  # 大型低层建筑
            'sparsely built',  # 稀疏建筑
            'heavy industry',  # 大型工厂
            'dense trees',  # 密集树木
            'scattered trees',  # 点型树木
            'bush& scrub',  # 灌木丛
            'low plants',  # 低矮植物
            'bare rock/ paved',  # 石头地
            'bare soil/ sand',  # 沙漠地
            'water'  # 水域
        ]


class RawVNRMSI(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(RawVNRMSI, self).__init__()
        self.class_list = ['building', 'cross', 'factory', 'farmland', 'highway', 'lake', 'river']
        assert split in self.class_list

        # N*256*256*3
        self.vnr = sio.loadmat(os.path.join(root, 'vnr_{}.mat'.format(split)))
        # N*256*256*4
        self.gf = sio.loadmat(os.path.join(root, 'gf_{}.mat'.format(split)))
        self.label = split

        self.transform = transform

    def __getitem__(self, item):
        vnr = self.vnr['data'][item].astype('int32')
        gf = self.gf['data'][item].astype('int32')
        data = (vnr, gf)
        label = self.class_list.index(self.label)

        if self.transform is not None:
            data, label = self.transform(data, label)
        return data, label

    def __len__(self):
        return self.vnr['data'].shape[0]

    @property
    def num_channels(self):
        return 3, 4

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(len(self.class_list)))

    @property
    def names(self):
        return self.class_list

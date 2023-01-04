import os
import pickle

from datas.base import Dataset
from datas.raw_dataset import RawSARMSI


class PreprocessedSARMSI(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(PreprocessedSARMSI, self).__init__()
        assert split in ['train', 'val', 'test']

        self.data_paths = [os.path.join(root, split, x) for x in os.listdir(os.path.join(root, split))]

        self.transform = transform

    def __getitem__(self, item):
        file = self.data_paths[item]
        fo = open(file, 'rb')
        data = pickle.load(fo)
        sen1 = data['data1']
        sen2 = data['data2']
        label = data['label']
        sens = (sen1, sen2)
        if self.transform is not None:
            sens, label = self.transform(sens, label)
        return sens, label

    def __len__(self):
        return len(self.data_paths)

    @property
    def num_channels(self):
        sens = self.__getitem__(0)[0]
        try:
            return sens.size()[0]
        except AttributeError:
            return sens[0].size()[0], sens[1].size()[0]

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(len(self.names)))

    @property
    def names(self):
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


class PreprocessedVNRMSI(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(PreprocessedVNRMSI, self).__init__()
        assert split in ['train', 'val', 'test']

        self.data_paths = [os.path.join(root, split, x) for x in os.listdir(os.path.join(root, split))]

        self.transform = transform

    def __getitem__(self, item):
        file = self.data_paths[item]
        fo = open(file, 'rb')
        data = pickle.load(fo)
        sen1 = data['data1']
        sen2 = data['data2']
        label = data['label']
        sens = (sen1, sen2)
        if self.transform is not None:
            sens, label = self.transform(sens, label)
        return sens, label

    def __len__(self):
        return len(self.data_paths)

    @property
    def num_channels(self):
        sens = self.__getitem__(0)[0]
        return sens.size()[0]

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(len(self.names)))

    @property
    def names(self):
        return [
            'building',
            'cross',
            'factory',
            'farmland',
            'highway',
            'lake',
            'river'
        ]

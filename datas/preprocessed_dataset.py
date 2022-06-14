import os
import pickle

from datas.base import Dataset


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
        return self.__getitem__(0)[0].size()[-1]

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(6))

    @property
    def names(self):
        return [
            'compact high-rise',  # 紧密型高层建筑
            'heavy industry',  # 大型工厂
            'dense trees',  # 密集树木
            'low plants',  # 低矮植物
            'bare rock/ paved',  # 石头地
            'water'  # 水域
        ]

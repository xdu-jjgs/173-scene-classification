import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawSARMSI


def build_transform():
    if CFG.DATASET.NAME == 'raw_sar_msi':
        class_interest = [0, 9, 10, 13, 14, 16]
        transform = transforms.Compose([
            transforms.SampleSelect(class_interest),
            transforms.LabelRenumber(class_interest),
            transforms.ToTensorPreData(),
            # TODO: measure
            transforms.NormalizePreData(mean=CFG.DATASET.MEANS, std=CFG.DATASET.STDS)
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'raw_sar_msi':
        dataset = RawSARMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for cropping'.format(CFG.DATASET.NAME))
    return dataset

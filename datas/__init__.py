import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawSARMSI


def build_transform():
    if CFG.DATASET.NAME == 'RAW_SAR_MSI':
        class_interest = CFG.DATASET.CLASSES_INTEREST
        transform = transforms.Compose([
            transforms.LabelFilter(class_interest),
            transforms.LabelRenumber(class_interest),
            transforms.ToTensorPreData(),
            transforms.NormalizePreData(
                means=[CFG.DATASET.SEN1.MEANS,CFG.DATASET.SEN2.MEANS],
                stds=[CFG.DATASET.SEN1.STDS, CFG.DATASET.SEN2.STDS]
            )
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'RAW_SAR_MSI':
        dataset = RawSARMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for cropping'.format(CFG.DATASET.NAME))
    return dataset

import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawSARMSI, RawVNRMSI
from datas.preprocessed_dataset import PreprocessedSARMSI, PreprocessedVNRMSI


def build_transform():
    if CFG.DATASET.NAME == 'RAW_SAR_MSI':
        class_interest = CFG.DATASET.CLASSES_INTEREST
        transform = transforms.Compose([
            transforms.LabelFilter(class_interest),
            transforms.LabelRenumber(class_interest),
            transforms.ToTensorPreData(),
            transforms.NormalizePreData(
                means=[CFG.DATASET.DATA1.MEANS, CFG.DATASET.DATA2.MEANS],
                stds=[CFG.DATASET.DATA1.STDS, CFG.DATASET.DATA2.STDS]
            )
        ])
    elif CFG.DATASET.NAME == 'RAW_VNR_MSI':
        # 用tensor+float格式存储比numpy+uint多出十几倍大小
        transform = transforms.Compose([
            transforms.ToTensorPreSubData(),
            transforms.NormalizePreData(
                means=[CFG.DATASET.DATA1.MEANS, CFG.DATASET.DATA2.MEANS],
                stds=[CFG.DATASET.DATA1.STDS, CFG.DATASET.DATA2.STDS]
            )
        ])
    elif CFG.DATASET.NAME in ['PREPROCESSED_SAR_MSI', 'PREPROCESSED_VNR_MSI', 'PREPROCESSED_VNR_MSI_extend']:
        if CFG.DATASET.FUSION == 'concat':
            transform = transforms.Compose([
                transforms.DataConcat()
            ])
        else:
            transform = None
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    # assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'RAW_SAR_MSI':
        dataset = RawSARMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    elif CFG.DATASET.NAME == 'RAW_VNR_MSI':
        dataset = RawVNRMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    elif CFG.DATASET.NAME == 'PREPROCESSED_SAR_MSI':
        dataset = PreprocessedSARMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    elif CFG.DATASET.NAME in ['PREPROCESSED_VNR_MSI', 'PREPROCESSED_VNR_MSI_extend']:
        dataset = PreprocessedVNRMSI(CFG.DATASET.ROOT, split, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for cropping'.format(CFG.DATASET.NAME))
    return dataset


def build_dataloader(dataset, split: str, sampler=None):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        return DataLoader(dataset,
                          batch_size=CFG.DATALOADER.BATCH_SIZE // dist.get_world_size(),
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler
                          )
    elif split == 'val':
        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler,
                          )
    elif split == 'test':
        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler
                          )

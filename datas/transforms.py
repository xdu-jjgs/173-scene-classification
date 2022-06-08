import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from typing import List


class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class RandomCrop(nn.Module):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, image, label):
        if image is not None:
            h, w, _ = image.shape
            new_h, new_w = self.size

            top = np.random.randint(0, h - new_h)
            down = top + new_h
            left = np.random.randint(0, w - new_w)
            right = left + new_w

            if image is not None:
                image = image[top:down, left:right, :]
            if label is not None:
                label = label[top:down, left:right]

        return image, label


class ToTensor(nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()
        # to C*H*W
        self.to_tensor = transforms.ToTensor()

    def forward(self, image, label):
        image = self.to_tensor(image)
        return image, label


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, image, label):
        image = self.normalize(image)
        return image, label


class ToTensorPreData(ToTensor):
    def __init__(self):
        super(ToTensorPreData, self).__init__()

    def forward(self, image, label):
        if image is not None:
            image = (super(ToTensorPreData, self).forward(sen, label)[0] for sen in image)
        return image, label


class NormalizePreData(Normalize):
    def __init__(self,mean, std):
        super(NormalizePreData, self).__init__(mean, std)

    def forward(self, image, label):
        if image is not None:
            image = (super(NormalizePreData, self).forward(sen, label)[0] for sen in image)
        return image, label


class SampleSelect(nn.Module):
    def __init__(self, class_interest, sample_num: int = -1):
        super(SampleSelect, self).__init__()
        self.class_interest = class_interest
        self.class_interest_count = [0] * len(class_interest)
        self.sample_num = sample_num

    def forward(self, image, label):
        if label in self.class_interest:
            if self.sample_num > 0:
                if self.class_interest_count[label] > self.sample_num:
                    return None, label
                else:
                    self.class_interest_count[label] += 1
            return image, label
        else:
            return None, label


class LabelRenumber(nn.Module):
    def __init__(self, class_interest):
        super(LabelRenumber, self).__init__()
        self.class_interest = class_interest

    def forward(self, image, label):
        if image is not None:
            label = self.class_interest.index(label)
        return image, label


class ChannelConcat(nn.Module):
    def __init__(self, class_interest):
        super(ChannelConcat, self).__init__()

    def forward(self, image, label):
        if image is not None:
            image = np.concatenate(image, axis=2)
            return image, label

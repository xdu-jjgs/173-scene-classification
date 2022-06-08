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
            if image is not None:
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


class ToTensorPreData(nn.Module):
    def __init__(self):
        super(ToTensorPreData, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image, label):
        if image is not None:
            image = [self.to_tensor(sen) for sen in image]
        return image, label


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, image, label):
        image = self.normalize(image)
        return image, label


class NormalizePreData(nn.Module):
    def __init__(self, means, stds):
        super(NormalizePreData, self).__init__()
        self.means = means
        self.stds = stds
        self.normalizes = [transforms.Normalize(mean, std) for mean, std in zip(self.means, self.stds)]

    def forward(self, image, label):
        image = [normalize(sen) for sen, normalize in zip(image, self.normalizes)]
        return image, label


class LabelFilter(nn.Module):
    def __init__(self, class_interest):
        super(LabelFilter, self).__init__()
        self.class_interest = class_interest

    def forward(self, image, label):
        if label in self.class_interest:
            return image, label
        else:
            return None, label


class LabelRenumber(nn.Module):
    def __init__(self, class_interest):
        super(LabelRenumber, self).__init__()
        self.class_interest = class_interest

    def forward(self, image, label):
        label = self.class_interest.index(label)
        return image, label


class ChannelConcat(nn.Module):
    def __init__(self):
        super(ChannelConcat, self).__init__()

    def forward(self, image, label):
        image = np.concatenate(image, axis=2)
        return image, label
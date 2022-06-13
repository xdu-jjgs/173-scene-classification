import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        n = input.shape[0]

        input = input.view(n, -1)
        target = target.view(n, -1)

        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-1 * ce_loss)
        # equals to:
        # pt = 1 / torch.exp(F.cross_entropy(input, target, reduction='none'))

        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        # equals to
        # loss = -1 * self.alpha * torch.pow(1 - pt, self.gamma) * torch.log(pt)
        return loss


def linear():
    raise NotImplementedError


def poly():
    raise NotImplementedError


def cosine(t, low, high):
    anneal_step = low - high
    return 0.5 * (1 + torch.cos(t / anneal_step * math.pi))


def build_anneal_function(function_name):
    function_name = function_name.lower()
    if function_name == 'linear':
        return linear
    elif function_name == 'poly':
        return poly
    elif function_name == 'cosine':
        return cosine


# TODO: Update anneal function
class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, anneal_function: str = 'cosine'):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.anneal_function = build_anneal_function(anneal_function)

    def forward(self, input, target):
        n = input.shape[0]

        input = input.view(n, -1)
        target = target.view(n, -1)
        pt = (1 - input) * target + input * (1 - target)
        step = self.anneal_function()
        focal_weight = (self.alpha * target + step * (1 - self.alpha) * (1 - target)) * pt.pow(
            self.gamma)

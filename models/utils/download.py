import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

models2urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def download_models(model_name: str, model_dir: str = './cache'):
    try:
        url = models2urls[model_name.lower()]
    except KeyError:
        raise KeyError("No matched link for {}!".format(model_name))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = model_zoo.load_url(url, model_dir=model_dir)
    return model


def load_pretrained_models(model: nn.Module, model_name: str) -> nn.Module:
    pretrained_dict = download_models(model_name)
    model.load_state_dict(pretrained_dict)
    return model

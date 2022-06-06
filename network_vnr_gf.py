# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:25:18 2020

@author: zn
"""
# import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from sklearn import preprocessing
# import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:1")
num_classification = 17


# ===============RAM_LSTM===================
class AttentionCropFunction(torch.autograd.Function):
    '''
    这部分用于实现apn求导
    代码来自RACNN https://github.com/jeong-tae/RACNN-pytorch
    '''

    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))  # given in paper k,now k=10
        in_size = images.size()[2]  # 假设=61 pytorch的数据格式[b,c,h,w]分别对应的是[batch,channal,height,width]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)  # 61*61 数组*数字=将其复制多少遍，将数组变成一个矩阵
        # x = torch.stack([unit.t()] * 3).type(torch.float32)  # 61*61*5 其中[61*61]的矩阵是整行从0一直到60，.t()是一个转置函数
        # y = torch.stack([unit] * 3).type(torch.float32)  # 61*61*5其中[61*61]的矩阵是整列从0一直到60。
        x = torch.stack([unit.t()] * 7).type(torch.float32)  # 改变通道数3==》18
        y = torch.stack([unit] * 7).type(torch.float32)  # 改变通道数3==》18
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []

        for i in range(images.size(0)):  # 表示每一个batch中的图像
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tx = tx if tx > (in_size / 3) else in_size / 3
            tx = tx if (in_size / 3 * 2) < tx else (in_size / 3 * 2)  # 这里的限制并不能封闭，且有一点矛盾
            ty = ty if ty > (in_size / 3) else in_size / 3
            ty = ty if (in_size / 3 * 2) < ty else (in_size / 3 * 2)
            tl = tl if tl > (in_size / 3) else in_size / 3  # tl至少为图像高或宽的1/3

            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size

            # print(w_off)
            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk
            xatt_cropped = xatt[:, h_off: h_end, w_off: w_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            # TODO 这里的size记得改
            xamp = F.upsample(before_upsample, size=(256, 256), mode='bilinear', align_corners=True)

            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        # TODO 这里的size记得改
        in_size = 256
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)

        #         show_image(inputs.cpu().data[0])
        #         show_image(ret_tensor.cpu().data[0])
        #         plt.imshow(norm[0].cpu().numpy(), cmap='gray')

        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 3 * 2)
        short_size = (in_size / 3)
        mx = (x >= long_size).float() - (x < short_size).float()  # 括号内的语句是判断语句，输出是0或1吗？
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


class CNNCell(nn.Module):
    # def _
    def __init__(self, input_bands, num_classification):
        super(CNNCell, self).__init__()
        # 输入batch,7,256,256
        '''
        一个简单的2D-CNN
        '''
        self.conv0 = nn.Conv2d(input_bands, 16, 3, 2, 1)
        self.bn0 = nn.BatchNorm2d(16)  # 128*128

        self.conv1 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)  # 64*64  输入apn

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)  # 32*32

        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)  # 16*16

        self.conv4 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)  # 8*8

        self.conv5 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(128)  # 4*4

        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(128)  # 2*2

        self.conv7 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn7 = nn.BatchNorm2d(256)  # 1*1
        self.fc1 = nn.Linear(256, num_classification)
        self.apn1 = nn.Sequential(
            # nn.Linear(32 * 7 * 7, 64),
            nn.Linear(32 * 64 * 64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )  # 【调参】

    def forward(self, x):
        # print('x shape', x.shape)
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        feature = x
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(-1, 256)  # 将x变成一个256列的矩阵
        # print("Shape",feature.shape)
        # box = self.apn1(feature.reshape(-1, 32 * 7 * 7))
        box = self.apn1(feature.reshape(-1, 32 * 64 * 64))

        x = self.fc1(x)
        return x, box


class RAM_DIFF(nn.Module):
    def __init__(self, input_bands, input_zise, num_classification=num_classification, step=3,
                 effective_step=[0, 1, 2]):
        super(RAM_DIFF, self).__init__()
        '''
        使用梯度的方法实现的RAM
        参数：
            input_bands ：输入波段数目
            input_zise：patch的窗口大小
            num_classification：类别数
            step：RNN的step数目
            effective_step：最后那些step的输出用于分类
        '''
        self.input_bands = input_bands
        self.input_zise = input_zise
        self.step = step
        self.num_classification = num_classification
        self.effective_step = effective_step
        self._all_layers = []
        self.cnncell = CNNCell(input_bands, num_classification)
        self.crop_resize = AttentionCropLayer()
        self.fc = nn.Linear(self.num_classification * len(self.effective_step), self.num_classification)

    def forward(self, input):
        x = input
        outputs = []
        for step in range(self.step):
            # print(input.shape)
            feature, box = self.cnncell(x)
            x = self.crop_resize(x, box * self.input_zise)
            if step in self.effective_step:
                outputs.append(feature)
        x = torch.cat(outputs, axis=-1)
        x = self.fc(x)
        return x
# ==============================================================


class discriminator(nn.Module):
    def __init__(self, num_classification):
        super(discriminator, self).__init__()
        self.num_classification = num_classification
        # TODO 这里参数记得改
        self.mainDNet = RAM_DIFF(input_bands=7, input_zise=256, num_classification=num_classification, step=3,
                                 effective_step=[0, 1, 2])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # 100,8,32,32 & 100,10,32,32 ==>100,18,32,32
        # print(x.shape)
        f = self.mainDNet(x)
        output = f.reshape(-1, self.num_classification)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)


if __name__ == '__main__':
    data1 = torch.randn([64, 8, 32, 32]).to(device)
    data2 = torch.randn([64, 10, 27, 32]).to(device)

    # G = generator().to(device)
    # x = G(noise, cluster_result)
    D = discriminator().to(device)
    # out_17c = D(data, cluster_result)
    out_17c = D(data1, data2, 1)
    # print('x', x.shape)
    print('out_17c', out_17c.shape)

import torch
import torch.nn as nn


class FuseNet(nn.Module):
    def __init__(self, num_classes, classifier1, classifier2):
        super(FuseNet, self).__init__()
        assert classifier1.out_channels == classifier2.out_channels
        out_channels = classifier1.out_channels // 2
        self.fuseblock1 = OctaveCB(in_channels=classifier1.out_channels, out_channels=out_channels, beta=0.8)
        self.fuseblock2 = OctaveCB(in_channels=out_channels, out_channels=out_channels, beta=0.8)
        self.fuseblock3 = OctaveCB(in_channels=out_channels, out_channels=out_channels, beta=0.8)
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.fc = nn.Linear(out_channels * 2, num_classes)

    def forward(self, x1, x2):
        x1, x2 = self.classifier1(x1), self.classifier2(x2)
        # print("x1 :{}, x2:{}".format(x1.size(), x2.size()))
        x1, x2 = self.fuseblock1(x1, x2)
        x1, x2 = self.fuseblock2(x1, x2)
        x1, x2 = self.fuseblock3(x1, x2)
        out = torch.cat((x1, x2), dim=1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.8, beta=0.8, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.h2h = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation, groups, bias)

        self.l2h = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation, groups, bias)
        self.l2l = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x1, x2):
        X_h2h = self.h2h(x1)
        X_h2l = self.h2l(x1)

        X_l2h = self.l2h(x2)
        X_l2l = self.l2l(x2)

        X_h = self.alpha * X_l2h + (1 - self.alpha) * X_h2h
        X_l = self.beta * X_h2l + (1 - self.beta) * X_l2l

        return X_h, X_l


class OctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.8, beta=0.8, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, beta, stride, padding, dilation, groups,
                               bias)
        self.bn_h = norm_layer(out_channels)
        self.bn_l = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x_h, x_l = self.conv(x1, x2)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, beta=0.8, stride=1, padding=1,
                 dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, beta, stride, padding,
                               dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels))
        self.bn_l = norm_layer(int(out_channels))

    def forward(self, x1, x2):
        x_h, x_l = self.conv(x1, x2)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l

import torch.nn as nn


class BRRNetAtrousConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BRRNetAtrousConv, self).__init__()

        self.atrous_convs = nn.ModuleList([])
        for i in range(6):
            self.atrous_convs.append(
                nn.Sequential(
                    # padding equals to dilation
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3,
                              padding=2 ** i, dilation=2 ** i),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        outputs = x = self.atrous_convs[0](x)
        for block in self.atrous_convs[1:]:
            # print(x.shape)
            x = block(x)
            outputs = outputs + x
        return outputs

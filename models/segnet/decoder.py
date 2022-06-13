import torch.nn as nn


class SegNetDecoder(nn.Module):
    def __init__(self, out_channels: int):
        super(SegNetDecoder, self).__init__()

        self.decoder = nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
            ),
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True),
            ),
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
            ),
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
            ),
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ),
        ])
        # print(*self.vgg16, sep='\n')

    def forward(self, x, max_pool_indexes):
        count_max_pools = 0
        for module in self.decoder:
            if isinstance(module, nn.MaxUnpool2d):
                x = module(x, max_pool_indexes[-1 * (count_max_pools + 1)])
                count_max_pools += 1
            else:
                x = module(x)
        return x

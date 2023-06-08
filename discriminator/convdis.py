import torch
import torch.nn as nn


class ConvDiscriminator(nn.Module):
    def __init__(self, hidden_size=16, img_size=32, in_channels=1):
        super(ConvDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.convers = nn.Sequential(
            ConvBlock(in_channels, hidden_size, bn=False),
            ConvBlock(hidden_size, hidden_size * 2),
            ConvBlock(hidden_size * 2, hidden_size * 4),
            ConvBlock(hidden_size * 4, hidden_size * 8),
        )
        downsample_size = img_size // 2 ** 4
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 8 * downsample_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        o = self.convers(x)
        o = o.view(o.shape[0], -1)
        return self.classifier(o)


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, bn=True):
        super(ConvBlock, self).__init__()
        self.bn = bn
        self.conver = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        )
        if bn:
            self.batchnorm = nn.BatchNorm2d(out_feat, 0.8)

    def forward(self, x):
        o = self.conver(x)
        o = self.batchnorm(o) if self.bn else o
        return o


if __name__ == "__main__":
    z = torch.rand((3, 3, 32, 32))
    d = ConvDiscriminator(in_channels=3)
    print(d(z))

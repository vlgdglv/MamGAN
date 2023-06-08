import torch
import torch.nn as nn
import torch.nn.functional as F
from auxiliary.resblock import ResBlock


class ResNetDiscriminator(nn.Module):
    def __init__(self, nf=64, num_channel=3):
        super(ResNetDiscriminator, self).__init__()
        self.conv = nn.Conv2d(num_channel, nf, kernel_size=(3, 3), stride=1, padding=1)

        self.resblocks = nn.ModuleList()
        for i in range(4):
            self.resblocks.append(ResBlock(nf * int(2 ** i), nf * int(2 ** (i + 1)), downsample=True))
        self.resblocks.append(ResBlock(nf * 16, nf * 16, downsample=True))

        self.linear = nn.Sequential(nn.Linear(nf * 16, 1), nn.Tanh())

    def forward(self, x):
        o = self.conv(x)
        for resblock in self.resblocks:
            o = resblock(o)
        o = o.sum(axis=(2, 3))
        o = self.linear(o)
        return o


if __name__ == "__main__":
    z = torch.rand((1, 3, 128, 128))
    rd = ResNetDiscriminator()
    print(rd(z).shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from auxiliary.resblock import ResBlock


class ResNetGenerator(nn.Module):
    def __init__(self, nf=64, dim_z=128, bottom_width=4, num_channel=3):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.linear = nn.Linear(dim_z, (bottom_width ** 2) * nf * 16)

        self.resblocks = nn.ModuleList()
        self.resblocks.append(ResBlock(nf * 16, nf * 16, upsample=True))
        for i in range(4):
            self.resblocks.append(ResBlock(nf * int(16 / (2 ** i)), nf * int(16 / (2 ** (i+1))), upsample=True))

        self.bn = nn.BatchNorm2d(nf)
        self.conv = nn.Conv2d(nf, num_channel, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        o = self.linear(x)
        o = o.view(o.shape[0], -1, self.bottom_width, self.bottom_width)

        for resblock in self.resblocks:
            o = resblock(o)

        o = F.relu(self.bn(o))
        o = F.tanh(self.conv(o))

        return o


if __name__ == "__main__":
    z = torch.rand((1, 100))
    rg = ResNetGenerator(dim_z=100)
    rg = rg.to("cuda")
    z = z.to("cuda")
    print(rg(z).shape)


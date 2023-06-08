import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_z=128, hidden_size=128, img_size=32):
        super(Generator, self).__init__()
        out_size = img_size * img_size
        self.img_size = img_size
        self.layers = nn.Sequential(
            Block(dim_z, hidden_size, bn=False),
            Block(hidden_size, hidden_size * 2),
            Block(hidden_size * 2, hidden_size * 4),
            Block(hidden_size * 4, hidden_size * 8),
            nn.Linear(hidden_size * 8, out_size),
            nn.Tanh()
        )

    def forward(self, x):
        o = self.layers(x)
        return o.view(o.shape[0], -1, self.img_size, self.img_size)


class Block(nn.Module):
    def __init__(self, in_feat, out_feat, bn=True):
        super(Block, self).__init__()
        self.norm = bn
        self.l1 = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(out_feat, 0.8)
        self.lr = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        o = self.l1(x)
        o = self.bn(o) if self.norm else o
        o = self.lr(o)
        return o


if __name__ == "__main__":
    z = torch.rand((3, 128))
    g = Generator()
    print(g(z))

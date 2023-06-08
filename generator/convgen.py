import torch
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, dim_z=128, hidden_size=128, img_size=32, out_channels=1):
        super(ConvGenerator, self).__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.head_size = img_size // 4

        self.linear = nn.Linear(dim_z, self.hidden_size * self.head_size ** 2)
        self.conver = nn.Sequential(
            nn.BatchNorm2d(self.hidden_size),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hidden_size, self.hidden_size // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_size // 2, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        o = self.linear(x)
        o = o.view(o.shape[0], self.hidden_size, self.head_size, self.head_size)
        o = self.conver(o)
        return o


if __name__ == "__main__":
    dim_z = 100
    z = torch.rand((3, dim_z))
    g = ConvGenerator(dim_z=dim_z)
    print(g(z).shape)

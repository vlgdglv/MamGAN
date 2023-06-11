import torch
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, dim_z=128, hidden_size=128, img_size=32, out_channels=1):
        super(ConvGenerator, self).__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.conver = nn.Sequential(
            nn.ConvTranspose2d(dim_z, hidden_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.hidden_size * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_size * 2, hidden_size * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_size, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        o = self.conver(x)
        return o


if __name__ == "__main__":
    dim_z = 100
    z = torch.rand((3, dim_z, 1, 1))
    g = ConvGenerator(dim_z=dim_z)
    print(g)
    print(g(z).shape)


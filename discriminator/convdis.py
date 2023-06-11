import torch
import torch.nn as nn


class ConvDiscriminator(nn.Module):
    def __init__(self, hidden_size=16, img_size=32, in_channels=1):
        super(ConvDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.convers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        o = self.convers(x)
        return o.view(-1)


if __name__ == "__main__":
    z = torch.rand((3, 3, 32, 32))
    d = ConvDiscriminator(in_channels=3)
    print(d(z))

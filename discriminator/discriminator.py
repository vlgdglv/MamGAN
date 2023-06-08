import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nf=128, hidden_size=256):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(nf, hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)


if __name__ == "__main__":
    z = torch.rand((3, 32, 4))
    d = Discriminator()
    print(d(z))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channles=None,
                 kernel_size=3, pad=1, upsample=False, downsample=False):
        super(ResBlock, self).__init__()
        hidden_channles = out_channels if hidden_channles is None else hidden_channles
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channles, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channles)
        self.conv2 = nn.Conv2d(hidden_channles, out_channels, kernel_size, padding=pad, bias=False)

        if upsample and downsample:
            upsample, downsample = False, False
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            self.upsampler = nn.UpsamplingNearest2d(scale_factor=2,)
        if downsample:
            self.downsampler = nn.AvgPool2d(kernel_size=(2, 2))
        self.learnable_sc = upsample or (in_channels != out_channels)
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_sc = None

    def forward(self, x):

        res = self.shortcut(x)
        o = F.relu(self.bn1(x))
        o = self.conv1(self.upsampler(o)) if self.upsample else self.conv1(o)
        o = self.conv2(F.relu(self.bn2(o)))
        o = self.downsampler(o) if self.downsample else o

        return o + res

    def shortcut(self, x):
        if self.conv_sc is not None:
            x = self.conv_sc(self.upsampler(x)) if self.upsample else self.conv_sc(x)
        x = self.downsampler(x) if self.downsample else x
        return x


if __name__ == "__main__":
    # N C HW
    x = torch.rand((1, 3, 31, 31))
    rb = ResBlock(in_channels=3, out_channels=64)
    print(rb(x).shape)

    rb1 = ResBlock(in_channels=3, out_channels=64, upsample=True)
    print(rb1(x).shape)

    rb2 = ResBlock(in_channels=3, out_channels=64, downsample=True)
    print(rb2(x).shape)



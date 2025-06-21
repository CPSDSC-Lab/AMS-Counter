from torch import nn


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, flag=True):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if flag:
            self.gn = nn.GroupNorm(8, out_channels)
        else:
            self.gn = nn.GroupNorm(1, out_channels)
        self.gelu = nn.GELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.flag = flag

    def forward(self, trg):
        trg = self.conv(trg)
        if self.flag:
            trg = self.up(self.gelu(self.gn(trg)))
        return trg
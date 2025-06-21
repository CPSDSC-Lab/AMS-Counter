import math

import einops
import torch
from torch import nn
import torch.nn.functional as F


def d3_to_d4(t):
    b, hw, c = t.size()
    if hw % 2 != 0:
        t = t[:, 1:]
    h = w = int(math.sqrt(hw))
    return t.transpose(1, 2).reshape(b, c, h, w)


class DensityDecoderDot(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.n_levels = 5
        convs = []
        current_dim = 512  # number of feature channels
        for i in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(current_dim, current_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, current_dim // 2),
                nn.GELU()
            )
            convs.append(decode_head)
            current_dim = current_dim // 2

        self.convs = nn.ModuleList(convs)

        self.final_conv = nn.Sequential(
            nn.Conv2d(current_dim, 1, kernel_size=1, stride=1)
        )
        # initialize weights
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x, y_, feature_maps, sim_map):
        sim_map = sim_map
        x = d3_to_d4(x)
        x = x * sim_map
        for i in range(self.n_levels):
            x = self.convs[i](x)
            if i < self.n_levels - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size=[384, 384], mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

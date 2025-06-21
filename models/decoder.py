import einops
import torch
from torch import nn
import torch.nn.functional as F


class DensityDecoder(nn.Module):
    """
    点密度图回归模块
    """

    def __init__(self, in_dim: int, target_hw: int, use_hierarchy: bool = False) -> None:
        super().__init__()

        self.n_levels = 4 if use_hierarchy else 2
        self.target_hw = [target_hw, target_hw]
        self.alpha = nn.Parameter(torch.tensor(0.5))
        convs = []
        current_dim = in_dim  # number of feature channels
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
        self.pyradim_conv = None  # the conv to squeeze the fine multimodel features
        if use_hierarchy:
            self.pyradim_conv = nn.Sequential(
                nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, stride=1),
                nn.GroupNorm(8, in_dim // 2),
                nn.GELU()
            )

    def forward(self, x):
        for i in range(self.n_levels):
            x = self.convs[i](x)
            if i < self.n_levels - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size=self.target_hw, mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

    def forward_hierarchical(self, xs):
        """
        xs: [14,14,512], [28,28,512]
        """
        x0, x1 = xs[0], xs[1]
        x = x0
        for i in range(self.n_levels):
            if i == 1:
                # x = x + self.pyradim_conv(x1)
                x = self.alpha * x + (torch.tensor(1) - self.alpha) * self.pyradim_conv(x1)
                x = x * torch.tensor(2.0)

            x = self.convs[i](x)
            if i < self.n_levels - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        x = torch.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

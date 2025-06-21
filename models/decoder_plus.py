import math

import einops
import torch
from torch import nn

from models.UpConv import UpConv
import torch.nn.functional as F

from models.models_crossvit import CrossAttentionBlock


def d3_to_d4(t):
    b, hw, c = t.size()
    if hw % 2 != 0:
        t = t[:, 1:]
    h = w = int(math.sqrt(hw))
    return t.transpose(1, 2).reshape(b, c, h, w)


class DensityDecoderPlus(nn.Module):
    def __init__(self):
        super().__init__()
        proj_dims = 64
        self.clip_out_dim = 512
        self.target_hw = [384, 384]

        self.density_decoder = nn.ModuleList([
            UpConv(proj_dims + 1, proj_dims, 3, 1),
            UpConv(proj_dims, proj_dims, 3, 1),
            UpConv(proj_dims, proj_dims, 3, 1),
            UpConv(proj_dims, proj_dims, 3, 1),
            UpConv(proj_dims, 1, 1, flag=False)
        ])

        self.cross_attention = nn.ModuleList([
            CrossAttentionBlock(self.clip_out_dim, 8, 4., qkv_bias=True, qk_scale=None,
                                norm_layer=nn.LayerNorm, drop=0.1, drop_path=0.1),
            CrossAttentionBlock(self.clip_out_dim, 8, 4., qkv_bias=True, qk_scale=None,
                                norm_layer=nn.LayerNorm, drop=0.1, drop_path=0.1)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(512, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(512, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(512, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )

        self.v_proj = nn.Linear(512, proj_dims, bias=True)
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=1, stride=1)
        # )

    def forward(self, patch_embedding, text_embedding, feature_maps, sim_map):
        # sim_map = F.cosine_similarity(patch_embedding.reshape(-1, 16, 16, patch_embedding.shape[-1]),
        #                               text_embedding.unsqueeze(1).expand(-1, 16, 16, -1), dim=-1)

        # sim_map = sim_map.unsqueeze(1)
        img_feat_patches_cross = self.cross_attention[0](patch_embedding, text_embedding)
        x = torch.cat([d3_to_d4(self.v_proj(img_feat_patches_cross)), sim_map], dim=1)
        # Density map regression
        for i, d in enumerate(self.density_decoder):
            if i == 1:
                x = d(x + self.proj(d3_to_d4(self.cross_attention[i](feature_maps[:, -1, :, :], text_embedding)))
                      * F.interpolate(sim_map, scale_factor=2))
            elif i == 2:
                x = d(x + self.proj1(d3_to_d4(feature_maps[:, -2, :, :])) * F.interpolate(sim_map, scale_factor=4))
            elif i == 3:
                x = d(x + self.proj2(d3_to_d4(feature_maps[:, -3, :, :])) * F.interpolate(sim_map, scale_factor=8))
            else:
                x = d(x)

        # x = self.final_conv(x)
        x = F.interpolate(x, size=self.target_hw, mode='bilinear', align_corners=False)
        x = torch.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')

        return x

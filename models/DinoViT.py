import functools
import math
import operator

import einops
import torch
from torch import nn


class DinoViT(nn.Module):
    """
    ViT encoder for CLIP
    """

    def __init__(self,
                 dino_model,
                 clip_embed_dim: int,
                 clip_out_dim: int,
                 use_vpt: bool,
                 vpt_width: int,
                 vpt_depth: int = 8,
                 unfreeze: bool = False) -> None:
        """
        Param:
            clip_model: pretrained OpenAI CLIP model
            use_vpt: whether to use visual prompt tuning
            vpt_width: number of vpt token per layer
            vpt_depth: number of vpt layers. 1: vpt at the first layer (shallow), >1: deep vpt
            unfreeze: If true, unfreeze the CLIP model
        """
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.clip_out_dim = clip_out_dim
        self.vit = dino_model
        if unfreeze:
            for param in self.vit.parameters():
                param.requires_grad = True
        self.use_vpt = use_vpt
        self.visual_prompt = None
        self.vpt_dropout = None
        self.vpt_norm = None
        self.vpt_proj = None
        self.vpt_depth = vpt_depth
        self.vpt_width = vpt_width
        self.visual_prompt = None
        self.text_proj = nn.Linear(clip_out_dim, clip_embed_dim)
        nn.init.kaiming_normal_(self.text_proj.weight, a=0, mode='fan_out')
        self.text_dropout = nn.Dropout(0.1)
        if use_vpt:
            self.vpt_dropout = nn.Dropout(0.1)
            self.vpt_norm = nn.LayerNorm(clip_embed_dim, eps=1e-6)
            self.vpt_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
            nn.init.kaiming_normal_(self.vpt_proj.weight, a=0, mode='fan_out')

            patch_size = self.vit.patch_size
            val = math.sqrt(
                6. / float(3 * functools.reduce(operator.mul, (patch_size, patch_size), 1) + self.clip_embed_dim))
            vpt = torch.empty((vpt_depth, vpt_width, clip_embed_dim))
            nn.init.uniform_(vpt, -val, val)
            self.visual_prompt = nn.Parameter(vpt)

    def forward(self, image, text_embedding):
        """
        input: image: [B, 3, 224, 224]
        text_embedding: [B, 1, 512]
        """
        x = self.vit.prepare_tokens_with_masks(image, None)
        # TODO: semantics conditions
        if self.use_vpt:
            vpts = einops.repeat(self.visual_prompt[0, ...], 'n d -> b n d', b=x.shape[0])
            spts = self.text_proj(text_embedding)
            self.spts = einops.repeat(spts, 'b 1 d -> b n d', n=20)
            x = torch.cat([x[:, :1, :],
                           self.vpt_dropout(self.vpt_proj(vpts)),
                           x[:, 1:, :]], dim=1)  # shape = [*, grid ** 2 + 1 + n_vpt + n_reg, width]

        # for blk in self.blocks:
        #     x = blk(x)
        layer_feats = []
        for i, blk in enumerate(self.vit.blocks):
            if i == 0:
                x = blk(x)
                layer_feats.append(x[:, (1 + self.vpt_width + self.vit.num_register_tokens):, :])
            elif i < self.vpt_depth:
                deep_prompt_emb = self.vpt_dropout(
                    self.vpt_proj(self.visual_prompt[i - 1, ...]).expand(x.shape[0], -1, -1)
                )
                x = torch.cat(
                    (
                        x[:, :1, :],
                        deep_prompt_emb,
                        x[:, (1 + self.vpt_width):, :],
                    ),
                    dim=1,
                )
                x = blk(x)
                layer_feats.append(x[:, (1 + self.vpt_width + self.vit.num_register_tokens):, :])
            elif i == self.vpt_depth:
                x = torch.cat((x[:, :1, :], x[:, (1 + self.vpt_width):, :]), dim=1)
                x = blk(x)
                layer_feats.append(x[:, (1 + self.vit.num_register_tokens):, :])
            else:
                x = blk(x)
                layer_feats.append(x[:, (1 + self.vit.num_register_tokens):, :])

        x_norm = self.vit.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, self.vit.num_register_tokens + 1:],
            "x_prenorm": x,
            "layer_feats": layer_feats
        }

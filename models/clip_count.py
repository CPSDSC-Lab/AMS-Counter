import math

import torch
import torch.nn as nn

import clip
from torchvision import transforms
import einops

from models.DinoViT import DinoViT
from models.SimFusion import SimFusion

from models.decoder import DensityDecoder
from models.decoder_plus import DensityDecoderPlus
from models.mlp import MLP
from models.models_crossvit import ConvCrossAttentionBlock, CrossAttentionBlock
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed


class CLIPCount(nn.Module):
    def __init__(self, fim_depth: int = 4,
                 fim_num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 norm_layer=nn.LayerNorm,
                 use_vpt: bool = True,
                 vpt_width: int = 2,
                 vpt_depth: int = 2,
                 use_coop: bool = True,
                 coop_width: int = 2,
                 backbone: str = "b16",
                 use_fim: bool = True,
                 use_mixed_fim: bool = False,
                 unfreeze_vit: bool = False):
        """
        The CLIP-Count models
        Param:
            fim_depth: the number of blocks for the patch-text interaction module, only useful for naive ViT.
            fim_num_heads: the number of heads for the patch-text interaction module.
            mlp_ratio: the ratio (mlp width)/(cross attn hidden dim) for the patch-text interaction module.
            norm_layer: the normalization layer for the patch-text interaction module.
            use_vpt: whether to use visual prompt tuning
            vpt_width: how much visual token used per layer,
            vpt_depth: how many layers used for visual prompt tuning (try allocate from the input layer first)
            use_coop: whether use coop for context learning.
            backbone: visual backbone of clip.
            use_fim: whether to use a naive transformer for patch-text interaction
            use_mixed_fim: whether to use a hierarchical transformer for patch-text interaction
            unfreeze_vit: whether to finetune all clip vit parameters.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if backbone == "b16":
            self.clip, clip_preprocess = clip.load("ViT-B/16")
            self.vit = torch.hub.load(r'./dinov2', 'dinov2_vits14_reg', source='local')
            self.n_patches = 16 * 16
            self.clip_hidden_dim = 384
            self.clip_out_dim = 512
        elif backbone == "b14":
            self.clip, clip_preprocess = clip.load("ViT-B/32")
            delattr(self.clip, 'visual')
            self.vit = torch.hub.load(r'./dinov2', 'dinov2_vitb14_reg', source='local')
            self.n_patches = 16 * 16
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "l14":
            self.clip, clip_preprocess = clip.load("ViT-L/14")
            del self.clip.visual
            self.vit = torch.hub.load(r'./dinov2', 'dinov2_vitl14_reg', source='local')
            self.n_patches = 16 * 16
            self.clip_hidden_dim = 1024
            self.clip_out_dim = 768

        self.clip = self.clip.to('cuda')
        self.vit = self.vit.to('cuda')
        self.clip.requires_grad_(False)
        self.vit.requires_grad_(False)

        self.use_coop = use_coop
        self.coop_width = coop_width

        self.use_fim = use_fim
        self.use_mixed_fim = use_mixed_fim
        # cannot use mixed_fim and fim at the same time
        assert (not use_fim) or (
            not use_mixed_fim), "You can not use hierachical transformer and plain transformer at the same time!"
        self.fim_blocks = None
        if use_mixed_fim:
            self.fim_blocks = nn.ModuleList([
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                        norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution=1.),
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                        norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution=2.),
            ])
            self.clip_layers = nn.ModuleList([
                CrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                    norm_layer=norm_layer, drop=0.1, drop_path=0.1),
                CrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                    norm_layer=norm_layer, drop=0.1, drop_path=0.1)
            ]).requires_grad_()

        elif use_fim:
            self.fim_blocks = nn.ModuleList([
                CrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                    norm_layer=norm_layer, drop=0.1, drop_path=0.1) for _ in range(fim_depth)
            ])

        self.decoder_norm = norm_layer(self.clip_out_dim)

        self.visual_project = MLP(self.clip_hidden_dim, [self.clip_hidden_dim // 4], self.clip_out_dim).to('cuda')
        self.text_project = MLP(self.clip_out_dim, [self.clip_out_dim // 4], self.clip_out_dim).to('cuda')

        # the PE for the patch embeddings \mathcal{E}_p
        # fixed sin-cos embedding
        n_token = self.n_patches
        self.patch_emb_pos_embed = nn.Parameter(torch.zeros(1, n_token, self.clip_out_dim), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.clip_out_dim, int(n_token ** 0.5), cls_token=False)
        self.patch_emb_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.Normalize(
                                                  mean=(0.48145466, 0.4578275, 0.40821073),
                                                  std=(0.26862954, 0.26130258, 0.27577711)
                                              )])

        self.img_encoder = DinoViT(self.vit, self.clip_hidden_dim, self.clip_out_dim, True, vpt_width, vpt_depth)
        self.text_encoder = CLIPTextTransformer(self.clip, use_coop=self.use_coop, n_ctx=self.coop_width)
        # --------------------------------------------------------------------------
        # CNN-based density decoder
        self.density_decoder = DensityDecoder(self.clip_out_dim, 384, use_hierarchy=use_mixed_fim)
        self.density_decoder = DensityDecoderPlus()
        # self.density_decoder = DensityDecoderDot()
        self.similarity_fusion = SimFusion()
        # --------------------------------------------------------------------------

    def forward_visual_encoder(self, x, y):
        """
        input: x: images, [B, 3, 384, 384]
        """
        # embed patches
        x = self.preprocess(x)
        features_dict = self.img_encoder(x, y)
        features = features_dict['x_norm_patchtokens']
        token = features_dict['x_norm_clstoken']
        feature_maps = features_dict["layer_feats"]
        feature_maps = [feature_maps[i] for i in [2, 5, 8]]
        feature_maps = torch.stack(feature_maps)
        feature_maps = feature_maps.permute(1, 0, 2, 3)

        return token, features, feature_maps

    def forward_decoder(self, img_feat_patches, text_embedding, cls_token, feature_maps):
        """

        """

        extra_out = {}

        x_cls = cls_token
        extra_out['x_cls'] = x_cls
        extra_out['text_embedding'] = text_embedding

        x = img_feat_patches
        x = self.visual_project(x)
        extra_out['patch_embedding'] = x

        x = x + self.patch_emb_pos_embed  # [B, n_tokens, 512]
        y_ = text_embedding  # [B, 1, 512]
        y_ = self.text_project(y_)

        feature_maps = self.visual_project(feature_maps)
        sim_map = self.similarity_fusion(x, feature_maps, text_embedding)

        extra_out['sim_map'] = sim_map
        extra_out['sim_map_16x'] = F.interpolate(sim_map, scale_factor=16)
        # 密度图回归
        if self.use_mixed_fim:
            # pred_density = self.density_decoder.forward_hierarchical(xs)
            pred_density = self.density_decoder(x, y_, feature_maps, sim_map)
        else:
            x = self.seq_2_2d(x)  # [B, 512, patch_w, patch_h]
            extra_out['pixel_text_matching_map'] = x
            pred_density = self.density_decoder(x)

        return pred_density, extra_out

    def forward(self, imgs, text, return_extra=True, coop_require_grad: bool = False):

        text_token = clip.tokenize(text).to(imgs.device)

        if coop_require_grad:
            text_embedding = self.text_encoder(text_token).float()
        else:
            with torch.no_grad():
                text_embedding = self.text_encoder(text_token).float()

        cls_token, img_feat_patches, feature_maps = self.forward_visual_encoder(imgs, text_embedding)

        pred_density, extra_out = self.forward_decoder(img_feat_patches, text_embedding, cls_token,
                                                       feature_maps)  # [N, 384, 384]

        if return_extra:
            return pred_density, extra_out
        return pred_density

    def seq_2_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x


class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """

    def __init__(self, clip_model, use_coop: bool, n_ctx: int = 2) -> None:
        super().__init__()

        self.clip_model = clip_model
        self.learnable_context = None
        self.use_coop = use_coop  # global context for all classes
        if use_coop:
            self.n_ctx = n_ctx
            context_vectors = torch.empty(self.n_ctx, self.clip_model.ln_final.weight.shape[0])
            torch.nn.init.normal_(context_vectors, std=.02)
            self.learnable_context = nn.Parameter(context_vectors)  # [n_ctx, 512]

    def forward(self, text):
        """
        Input:
            text: tokenized text, shape = [batch_size, n_ctx]
        """
        x = self.clip_model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if self.use_coop:
            sos_token = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, d_model]
            suffix_tokens = x[:, 1:-self.n_ctx, :]  # class tokens + [EOS] token
            ctx = einops.repeat(self.learnable_context, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([sos_token, ctx, suffix_tokens], dim=1)

        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        x = x.unsqueeze(1)  # [batch_size, 1, transformer.width]
        return x


if __name__ == "__main__":
    clip_count = CLIPCount()

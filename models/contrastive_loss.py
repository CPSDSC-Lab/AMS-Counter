import torch
import torch.nn as nn
import torch.nn.functional as F

from util.init_weights import trunc_normal_init, constant_init


class ContrastiveLoss(nn.Module):
    def __init__(self, clip_encoder, temperature=0.07, noise_text_ratio=0.0, normalize=False):
        super().__init__()
        self.temperature = temperature
        self.noise_text_ratio = noise_text_ratio
        self.normalize = normalize
        self.text_encoder = clip_encoder

    def forward(self, patch_embedding, img_embedding, gt_text_embedding, noise_text_embeddings, gt_density, prompt_gt,
                sim_map):
        """
        Args:
            patch_embedding: (B, 196, 512) embedding of image patch feature
            img_embedding: (B, 1, 512) embedding of image feature
            gt_text_embedding: (B, 1, 512), ground truth text embedding
            noise_text_embeddings: (N, 1, 512), noise text embeddings
            gt_density: (B, 384, 384), ground truth density map
        """
        gt_density = F.interpolate(gt_density.unsqueeze_(1), size=(224, 224), mode='nearest')

        density_mask = F.max_pool2d(gt_density, kernel_size=14, stride=14, padding=0)  # same as ViT conv1
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3, 1)  # (B, 16, 16, 1)
        gt_text_embedding = gt_text_embedding.unsqueeze(1).expand(-1, 16, 16, -1)

        # [B, 14, 14, 512], contains both gt and noise text embedding
        fused_text_embedding_map = gt_text_embedding
        pos_mask = density_mask.squeeze_(-1)  # (B, 14, 14, 1)

        patch_embeddings = patch_embedding.reshape(-1, 16, 16, patch_embedding.shape[-1])
        # batch cosine similarity, this function automatically normalizes the vectors
        # sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map, dim=-1)  # (B, 14, 14)
        sim_map = sim_map.squeeze(1)
        # sim_global = F.cosine_similarity(img_embedding, fused_text_embedding_map , dim=-1) # (B, 1)
        n_pos = torch.sum(pos_mask, dim=(1, 2))  # (B) how many positive samples in each batch
        # if n_pos == 0, set to 1 to avoid nan
        n_pos = torch.where(n_pos == 0, torch.ones_like(n_pos), n_pos)
        # infoNCE
        sim_map = torch.exp(sim_map / self.temperature)
        pos_sum = torch.sum(torch.where(pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5
        neg_sum = torch.sum(torch.where(~pos_mask, sim_map, torch.zeros_like(sim_map)), dim=(1, 2)) + 1e-5

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        if self.normalize:
            loss = loss / n_pos
        return loss.mean()

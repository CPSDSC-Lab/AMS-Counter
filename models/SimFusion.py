import torch
from torch import nn, cosine_similarity

from models.UFFC import FourierUnit_modified


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SimFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.init.xavier_normal_(self.visual_proj.weight)

        self.fourier = FourierUnit_modified(in_channels=3, out_channels=3)
        self.weights = nn.Parameter(torch.ones(16, 16), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(16, 16), requires_grad=True)

        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, final_map, assi_map, text_embedding):
        B = final_map.shape[0]

        text_embedding = text_embedding.unsqueeze(1).expand(-1, 16, 16, -1)

        patch_embeddings = final_map.reshape(-1, 16, 16, 512)
        assi_embeddings = assi_map.reshape(-1, 3, 16, 16, 512)

        assi_sim_map = cosine_similarity(assi_embeddings, text_embedding.unsqueeze(1).expand(-1, 3, -1, -1, -1), dim=-1)
        sim_map = cosine_similarity(patch_embeddings, text_embedding, dim=-1)

        assi_sim_map = self.fourier(assi_sim_map)
        sim_map = self.weights * sim_map + self.bias

        sim_map_fusion = torch.concat([assi_sim_map, sim_map.unsqueeze(1)], dim=1)
        sim_map_fusion = self.conv(sim_map_fusion)

        sim_final = sim_map.unsqueeze(1) * self.alpha + sim_map_fusion * (1.0 - self.alpha)

        return sim_final


if __name__ == '__main__':
    feature_maps = torch.randn(32, 3, 16 * 16, 512)
    text_embedding = torch.randn(32, 1, 512)
    final_map = torch.randn(32, 16 * 16, 512)

    sim_fusion = SimFusion()
    map = sim_fusion(final_map, feature_maps, text_embedding)
    a = 1

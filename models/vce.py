#VCE per frame visual reliability score
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalVCE(nn.Module):
    def __init__(self, visual_dim=512, audio_dim=64, hidden_dim=128, smooth_kernel=5):
        super().__init__()
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.aud_proj = nn.Linear(audio_dim, hidden_dim)
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        nn.init.zeros_(self.score_net[2].weight)
        nn.init.zeros_(self.score_net[2].bias)
        self.smooth_kernel = smooth_kernel
        if smooth_kernel > 1:
            self.smoother = nn.Conv1d(1, 1, kernel_size=smooth_kernel, padding=0, bias=False)
            nn.init.constant_(self.smoother.weight, 1.0 / smooth_kernel)
        else:
            self.smoother = None

    def forward(self, visual_feat, audio_feat):
        v = self.vis_proj(visual_feat)
        a = self.aud_proj(audio_feat)
        alpha = self.score_net(torch.cat([v, a], dim=-1))
        if self.smoother is not None:
            alpha = alpha.permute(0, 2, 1)
            alpha = F.pad(alpha, (self.smooth_kernel - 1, 0))
            alpha = self.smoother(alpha)
            alpha = alpha.permute(0, 2, 1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha

class VCEWithTemporalSmoothing(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=None, smooth_kernel=5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid())
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)
        self.smooth_kernel = smooth_kernel
        self.smoother = nn.Conv1d(1, 1, kernel_size=smooth_kernel, padding=0, bias=False)
        nn.init.constant_(self.smoother.weight, 1.0 / smooth_kernel)

    def forward(self, visual_feat):
        alpha = self.net(visual_feat)
        alpha = alpha.permute(0, 2, 1)
        alpha = F.pad(alpha, (self.smooth_kernel - 1, 0))
        alpha = self.smoother(alpha)
        alpha = alpha.permute(0, 2, 1)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha

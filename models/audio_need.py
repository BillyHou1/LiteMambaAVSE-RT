#AudioNeed estimator
#Reference: 
#https://github.com/ms-dot-k/AVSR
#https://github.com/anicolson/DeepXi
#https://ieeexplore.ieee.org/document/940795
import torch
import torch.nn as nn

class AudioNeedEstimator(nn.Module):
    def __init__(self, hidden_dim=16, warmup_steps=5000, n_freq=201, sample_rate=16000):
        super().__init__()
        self.warmup_steps = warmup_steps
        n_fft = (n_freq - 1) * 2
        hz = sample_rate / n_fft
        self.sp_lo = int(300.0 / hz)
        self.sp_hi = int(4000.0 / hz)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1))
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, -2.0)

    def _stats(self, mag):
        eps = 1e-8
        e = mag.pow(2).mean(dim=-1)
        nle = -torch.log(e.clamp(min=eps)) / 5.0
        log_mag = torch.log(mag.clamp(min=eps)).mean(dim=-1)
        geo = torch.exp(log_mag)
        sf = geo / mag.mean(dim=-1).clamp(min=eps)
        sb = mag[:, :, self.sp_lo:self.sp_hi]
        hr = sb.mean(dim=-1) / sb.max(dim=-1).values.clamp(min=eps)
        te = mag.pow(2).sum(dim=-1).clamp(min=eps)
        isr = 1.0 - sb.pow(2).sum(dim=-1) / te
        return torch.stack([nle, sf, hr, isr], dim=-1)

    def forward(self, noisy_mag, step=0):
        B, C, T, F = noisy_mag.shape
        if self.training and step < self.warmup_steps:
            return torch.zeros(B, T, 1, device=noisy_mag.device, dtype=noisy_mag.dtype)
        mag = noisy_mag.squeeze(1)
        return torch.sigmoid(self.mlp(self._stats(mag)))

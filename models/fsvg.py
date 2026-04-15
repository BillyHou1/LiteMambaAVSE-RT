#FSVG per frequency visual gate
#Reference: 
#https://github.com/johnarevalo/gmu-mmimdb
#https://github.com/hujie-frank/SENet
#https://github.com/prs-eth/FILM-Ensemble
#https://github.com/kagaminccino/LAVSE
import torch
import torch.nn as nn

class FSVG(nn.Module):
    def __init__(self, in_channels, hidden_channels=None,
                 context_kernel=3, use_interactions=True, alpha_channels=0):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(8, in_channels // 2)
        self.use_interactions = use_interactions
        self.alpha_channels = alpha_channels
        if use_interactions:
            fusion_ch = 4 * in_channels + alpha_channels
        else:
            fusion_ch = 2 * in_channels + alpha_channels
        self.gate_net = nn.Sequential(
            nn.Conv2d(fusion_ch, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=context_kernel,
                padding=context_kernel // 2, groups=hidden_channels, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True))
        final_conv = self.gate_net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, -1.0)

    def forward(self, audio_feat, visual_feat, alpha=None):
        if self.use_interactions:
            inputs = [audio_feat, visual_feat,
                      audio_feat - visual_feat,
                      audio_feat * visual_feat]
        else:
            inputs = [audio_feat, visual_feat]
        if alpha is not None:
            inputs.append(alpha)
        logits = self.gate_net(torch.cat(inputs, dim=1))
        return torch.sigmoid(logits)

#TF Mamba blocks, based on SEMamba
import torch
import torch.nn as nn
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.modules.mamba_simple import Block
except ImportError:
    from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm

def create_block(d_model, cfg, layer_idx=0):
    d_state = cfg['model_cfg']['d_state']
    d_conv = cfg['model_cfg']['d_conv']
    expand = cfg['model_cfg']['expand']
    norm_eps = cfg['model_cfg']['norm_epsilon']
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(RMSNorm, eps=norm_eps)
    block = Block(d_model, mixer_cls, mlp_cls=nn.Identity,
                  norm_cls=norm_cls, fused_add_norm=False,
                  residual_in_fp32=False)
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.forward_blocks = nn.ModuleList([create_block(in_channels, cfg)])
        if bidirectional:
            self.backward_blocks = nn.ModuleList([create_block(in_channels, cfg)])
        self.apply(partial(_init_weights, n_layer=1))

    def forward(self, x):
        xf = x.clone()
        rf = None
        for layer in self.forward_blocks:
            xf, rf = layer(xf, rf)
        yf = xf + rf if rf is not None else xf
        if not self.bidirectional:
            return yf
        xb = torch.flip(x, [1])
        rb = None
        for layer in self.backward_blocks:
            xb, rb = layer(xb, rb)
        yb = xb + rb if rb is not None else xb
        yb = torch.flip(yb, [1])
        return torch.cat([yf, yb], -1)

class TFMambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hid = cfg['model_cfg']['hid_feature']
        self.time_mamba = MambaBlock(self.hid, cfg, bidirectional=True)
        self.freq_mamba = MambaBlock(self.hid, cfg, bidirectional=True)
        self.tlinear = nn.ConvTranspose1d(self.hid * 2, self.hid, 1)
        self.flinear = nn.ConvTranspose1d(self.hid * 2, self.hid, 1)

    def forward(self, x):
        b, c, t, f = x.size()
        xt = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        xt_out = self.time_mamba(xt)
        xt_out = self.tlinear(xt_out.permute(0, 2, 1)).permute(0, 2, 1)
        xt = xt + xt_out
        xf = xt.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        xf_out = self.freq_mamba(xf)
        xf_out = self.flinear(xf_out.permute(0, 2, 1)).permute(0, 2, 1)
        xf = xf + xf_out
        return xf.view(b, t, f, c).permute(0, 3, 1, 2)

class CausalTFMambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hid = cfg['model_cfg']['hid_feature']
        self.time_mamba = MambaBlock(self.hid, cfg, bidirectional=False)
        self.freq_mamba = MambaBlock(self.hid, cfg, bidirectional=True)
        self.tlinear = nn.ConvTranspose1d(self.hid, self.hid, 1)
        self.flinear = nn.ConvTranspose1d(self.hid * 2, self.hid, 1)

    def forward(self, x):
        b, c, t, f = x.size()
        xt = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        xt_out = self.time_mamba(xt)
        xt_out = self.tlinear(xt_out.permute(0, 2, 1)).permute(0, 2, 1)
        xt = xt + xt_out
        xf = xt.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        xf_out = self.freq_mamba(xf)
        xf_out = self.flinear(xf_out.permute(0, 2, 1)).permute(0, 2, 1)
        xf = xf + xf_out
        return xf.view(b, t, f, c).permute(0, 3, 1, 2)

#ShuffleNetV2 visual encoder
#Reference: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def _channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, H, W)

class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super().__init__()
        self.benchmodel = benchmodel
        oup_inc = oup // 2
        if benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.banch2(x2)), 1)
        else:
            out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        return _channel_shuffle(out, 2)

class LiteTCN(nn.Module):
    def __init__(self, in_ch=512, bottleneck=64, dilations=(1, 2, 4)):
        super().__init__()
        self.compress = nn.Conv1d(in_ch, bottleneck, 1)
        self.tcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bottleneck, bottleneck, 5, dilation=d,
                          padding=2 * d, groups=bottleneck),
                nn.GroupNorm(4, bottleneck),
                nn.SiLU(),
            ) for d in dilations
        ])
        self.expand = nn.Conv1d(bottleneck, in_ch, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        h = self.compress(x)
        for layer in self.tcn_layers:
            h = h + layer(h)
        return x + self.expand(h)

class ShuffleNetVisualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vis_cfg = cfg.get('visual_cfg', {})
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(24),
            nn.PReLU(24),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        stage_repeats = [4, 8, 4]
        stage_ch = [24, 116, 232, 464]
        features = []
        in_ch = 24
        for idx in range(3):
            out_ch = stage_ch[idx + 1]
            for i in range(stage_repeats[idx]):
                if i == 0:
                    features.append(_InvertedResidual(in_ch, out_ch, 2, 2))
                else:
                    features.append(_InvertedResidual(in_ch, out_ch, 1, 1))
                in_ch = out_ch
        conv_last = nn.Sequential(
            nn.Conv2d(464, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))
        global_pool = nn.Sequential(nn.AvgPool2d(3))
        self.trunk = nn.Sequential(nn.Sequential(*features), conv_last, global_pool)

        self.channel_proj = nn.Conv1d(1024, 512, kernel_size=1)
        tcn_bottleneck = vis_cfg.get('tcn_bottleneck', 64)
        self.temporal_head = LiteTCN(in_ch=512, bottleneck=tcn_bottleneck, dilations=(1, 2, 4))
        self.temporal_shift = vis_cfg.get('visual_temporal_shift', 0)

        if vis_cfg.get('freeze_visual_encoder', True):
            for p in self.frontend3D.parameters():
                p.requires_grad = False
            for p in self.trunk.parameters():
                p.requires_grad = False

        pretrained_path = vis_cfg.get('lipreading_weights', None)
        if pretrained_path:
            self._load_weights(pretrained_path)

    def _load_weights(self, path):
        if not os.path.exists(path):
            print(f"pretrained weights not found: {path}")
            return
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        filtered = {}
        for k, v in state.items():
            if k.startswith('frontend3D.') or k.startswith('trunk.'):
                filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"loaded {len(filtered) - len(unexpected)}/{len(filtered)} pretrained params")

    def _preprocess(self, video):
        return 0.299 * video[:, 0:1] + 0.587 * video[:, 1:2] + 0.114 * video[:, 2:3]

    def train(self, mode=True):
        super().train(mode)
        for p in self.frontend3D.parameters():
            if not p.requires_grad:
                self.frontend3D.eval()
                break
        for p in self.trunk.parameters():
            if not p.requires_grad:
                self.trunk.eval()
                break
        return self

def _extract_spatial(self, video):
    B, C, Tv, H, W = video.shape
    gray = self._preprocess(video)
    
    if all(not p.requires_grad for p in self.frontend3D.parameters()):
        with torch.no_grad():
            x = self.frontend3D(gray)
    else:
        x = self.frontend3D(gray)
        
    _, C2, T2, H2, W2 = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T2, C2, H2, W2)
    
    if all(not p.requires_grad for p in self.trunk.parameters()):
        with torch.no_grad():
            x = self.trunk(x)
    else:
        x = self.trunk(x)
        
    return x.view(B, T2, -1).permute(0, 2, 1), T2
    
    def _extract_spatial(self, video):
        B, C, Tv, H, W = video.shape
        gray = self._preprocess(video)
        with torch.no_grad():
            x = self.frontend3D(gray)
        _, C2, T2, H2, W2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T2, C2, H2, W2)
        with torch.no_grad():
            x = self.trunk(x)
        return x.view(B, T2, -1).permute(0, 2, 1), T2

    def forward(self, video, T_audio):
        spatial, T2 = self._extract_spatial(video)
        features = self.channel_proj(spatial)
        if self.temporal_shift > 0 and features.shape[2] > self.temporal_shift:
            features = F.pad(features[:, :, :-self.temporal_shift],
                             (self.temporal_shift, 0))
        features = self.temporal_head(features)
        return F.interpolate(features, size=T_audio, mode="linear", align_corners=False)

def create_visual_encoder(cfg):
    return ShuffleNetVisualEncoder(cfg)

#SEMamba and LiteAVSEMamba generators
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba_block import TFMambaBlock, CausalTFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from .vce import CrossModalVCE, VCEWithTemporalSmoothing
from .fsvg import FSVG
from .lite_visual_encoder import create_visual_encoder
from .audio_need import AudioNeedEstimator

class SEMamba(nn.Module):
    def __init__(self, cfg):
        super(SEMamba, self).__init__()
        self.cfg = cfg
        n = cfg['model_cfg']['num_tfmamba']
        self.num_tscblocks = n if n is not None else 4
        self.dense_encoder = DenseEncoder(cfg)
        self.ts_mamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)
        x = torch.cat((noisy_mag, noisy_pha), dim=1)
        x = self.dense_encoder(x)
        for block in self.ts_mamba:
            x = block(x)
        mask = self.mask_decoder(x)
        denoised_mag = rearrange(mask * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1)
        return denoised_mag, denoised_pha, denoised_com

#LitAVSEMamba-RT
class LiteAVSEMamba(nn.Module):
    def __init__(self, cfg):
        super(LiteAVSEMamba, self).__init__()
        self.cfg = cfg
        self.hid = cfg['model_cfg']['hid_feature']
        self.use_visual = cfg.get('visual_cfg', {}).get('use_visual', True)
        lite_cfg = cfg.get('lite_cfg', {})
        self.vis_drop_rate = lite_cfg.get('visual_dropout_rate', 0.0)
        self.current_step = 0
        self.use_unified_gate = lite_cfg.get('use_unified_gate', True)
        self.use_cross_modal_vce = lite_cfg.get('use_cross_modal_vce', True)

        self.dense_encoder = DenseEncoder(cfg)
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

        n_blocks = cfg['model_cfg'].get('num_tfmamba', 4)
        self.ts_mamba = nn.ModuleList([CausalTFMambaBlock(cfg) for _ in range(n_blocks)])

        if self.use_visual:
            self.visual_encoder = create_visual_encoder(cfg)
            self.visual_early_proj = nn.Conv1d(512, self.hid, kernel_size=1)
            self.visual_channel_proj = nn.Conv2d(self.hid, 1, kernel_size=1)
            self.visual_proj = nn.Sequential(
                nn.Conv1d(512, self.hid, kernel_size=1),
                nn.GroupNorm(1, self.hid))
            self.visual_aux_head = nn.Linear(self.hid, self.hid)

            if self.use_unified_gate:
                n_freq = cfg['stft_cfg']['n_fft'] // 2 + 1
                self.audio_early_proj = nn.Linear(n_freq, self.hid)
                if self.use_cross_modal_vce:
                    self.vce = CrossModalVCE(
                        visual_dim=512, audio_dim=self.hid,
                        hidden_dim=128, smooth_kernel=5)
                else:
                    self.vce = VCEWithTemporalSmoothing(
                        input_dim=512, hidden_dims=[256, 64],
                        smooth_kernel=5)
                self.fsvg = FSVG(
                    in_channels=self.hid,
                    context_kernel=3, use_interactions=True,
                    alpha_channels=1)
                self.scout_mode = lite_cfg.get('scout_mode', 'none')
                self.scout_frames = lite_cfg.get('scout_frames', 2)
                self.use_audio_need = lite_cfg.get('use_audio_need', False)
                if self.use_audio_need:
                    self.audio_need = AudioNeedEstimator(
                        hidden_dim=16,
                        warmup_steps=lite_cfg.get('audio_need_warmup', 5000),
                        n_freq=n_freq,
                        sample_rate=cfg['stft_cfg']['sampling_rate'])
                    self.audio_need_alpha = lite_cfg.get('audio_need_alpha', 0.75)
            else:
                self.use_audio_need = False
                self.scout_mode = 'none'
                self.scout_frames = 0

    def _visual_dropout(self, vis, B):
        if not self.training or self.vis_drop_rate <= 0:
            return vis
        Tv = vis.shape[2]
        mask_len = max(2, int(Tv * self.vis_drop_rate))
        if Tv <= mask_len:
            return vis
        mask = torch.ones(B, 1, Tv, device=vis.device)
        starts = torch.randint(0, Tv - mask_len, (B,))
        for i in range(B):
            mask[i, :, starts[i]:starts[i] + mask_len] = 0.0
        return vis * mask

    def _scout(self, vis_raw):
        if self.scout_mode == 'none' or self.scout_frames <= 0:
            return vis_raw
        k = self.scout_frames
        v_pad = F.pad(vis_raw, (0, k), mode='replicate')
        return F.avg_pool1d(v_pad, kernel_size=k + 1, stride=1)

    def forward(self, noisy_mag, noisy_pha, video=None,
                return_intermediates=False, visual_degraded=None):
        intermediates = {}
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)

        if self.use_visual and video is not None:
            B = noisy_mag.shape[0]
            T = noisy_mag.shape[2]
            Freq = noisy_mag.shape[3]

            vis_raw = self.visual_encoder(video, T)
            vis_raw = self._visual_dropout(vis_raw, B)

            if self.use_unified_gate:
                scout_vis = self._scout(vis_raw)

                vis_for_vce = scout_vis.permute(0, 2, 1)
                audio_proj = self.audio_early_proj(noisy_mag.squeeze(1))
                if self.use_cross_modal_vce:
                    alpha = self.vce(vis_for_vce, audio_proj)
                else:
                    alpha = self.vce(vis_for_vce)

                intermediates['alpha'] = alpha.detach()
                intermediates['visual_raw'] = vis_raw.detach()

                if self.training and visual_degraded is not None:
                    gate_target = torch.where(
                        visual_degraded.bool(),
                        torch.tensor(0.15, device=alpha.device),
                        torch.tensor(0.85, device=alpha.device))
                    alpha_mean = alpha.mean(dim=1).squeeze(-1)
                    intermediates['gate_loss'] = F.binary_cross_entropy(alpha_mean, gate_target)

                if self.scout_mode == 'full':
                    vis_for_fusion = scout_vis
                else:
                    vis_for_fusion = vis_raw
                vis_proj = self.visual_early_proj(vis_for_fusion)
                vis_2d = vis_proj.unsqueeze(-1).expand(-1, -1, -1, Freq)

                audio_2d = audio_proj.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, Freq)
                alpha_feat = alpha.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, Freq)
                freq_gate = self.fsvg(audio_2d, vis_2d, alpha=alpha_feat)

                if return_intermediates:
                    intermediates['freq_gate'] = freq_gate.detach()

                alpha_bc = alpha.permute(0, 2, 1).unsqueeze(-1)
                if self.use_audio_need:
                    r_need = self.audio_need(noisy_mag, step=self.current_step)
                    need_amp = 1.0 + self.audio_need_alpha * r_need
                    need_amp_bc = need_amp.permute(0, 2, 1).unsqueeze(-1)
                    vis_gated = alpha_bc * freq_gate * need_amp_bc * vis_2d
                    if return_intermediates:
                        intermediates['r_need'] = r_need.detach()
                        intermediates['need_amplifier'] = need_amp.detach()
                else:
                    vis_gated = alpha_bc * freq_gate * vis_2d
            else:
                vis_proj = self.visual_early_proj(vis_raw)
                vis_gated = vis_proj.unsqueeze(-1).expand(-1, -1, -1, Freq)

            vis_ch = self.visual_channel_proj(vis_gated)
            x = torch.cat((noisy_mag, noisy_pha, vis_ch), dim=1)
            x = self.dense_encoder(x)
            B, C, T, Fenc = x.shape

            if self.training or return_intermediates:
                audio_target = x.mean(dim=-1).detach()
                vis_aux = self.visual_proj(vis_raw).permute(0, 2, 1)
                intermediates['aux_loss'] = F.mse_loss(
                    self.visual_aux_head(vis_aux), audio_target.permute(0, 2, 1))

        else:
            zero_ch = torch.zeros_like(noisy_mag)
            x = torch.cat((noisy_mag, noisy_pha, zero_ch), dim=1)
            x = self.dense_encoder(x)
            B, C, T, Fenc = x.shape

        for block in self.ts_mamba:
            x = block(x)
        mask = self.mask_decoder(x)
        denoised_mag = rearrange(mask * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1)

        if return_intermediates:
            return denoised_mag, denoised_pha, denoised_com, intermediates
        return denoised_mag, denoised_pha, denoised_com

import random
import logging
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data
from models.stfts import mag_phase_stft
from dataloaders.av_utils import load_video_frames, pad_or_trim_video, apply_visual_aug
from dataloaders.augment_rir import RIRAugmentor

logger = logging.getLogger(__name__)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

def mix_audio(clean, noise, snr_db):
    n = clean.size(0)
    if noise.size(0) >= n:
        start = random.randint(0, noise.size(0) - n)
        noise = noise[start:start + n]
    else:
        noise = noise.repeat(n // noise.size(0) + 1)[:n]
    clean_pow = torch.mean(clean ** 2) + 1e-10
    noise_pow = torch.mean(noise ** 2) + 1e-10
    scale = torch.sqrt(clean_pow / (10 ** (snr_db / 10.0) * noise_pow))
    return clean + scale * noise

class AVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json, cfg, split=True, visual_augmentation=False,
                 rir_json=None, rir_prob=0.3):
        if isinstance(data_json, str):
            self.data_list = load_json(data_json)
        else:
            self.data_list = data_json
        if noise_json:
            self.noise_paths = load_json(noise_json) if isinstance(noise_json, str) else noise_json
        else:
            self.noise_paths = []
        self.noise_cache = {}
        self.cfg = cfg
        self.split = split
        self.visual_augmentation = visual_augmentation
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.hop = cfg['stft_cfg']['hop_size']
        self.win = cfg['stft_cfg']['win_size']
        self.compress = cfg['model_cfg']['compress_factor']
        self.sr = cfg['stft_cfg'].get('sampling_rate', 16000)
        vis_cfg = cfg.get('visual_cfg', {})
        self.face_size = vis_cfg.get('face_size', 96)
        self.fps = vis_cfg.get('video_fps', 25)
        self.rir = RIRAugmentor(rir_json, rir_prob, self.sr) if rir_json else None

    def _get_noise(self, path):
        if path not in self.noise_cache:
            data, _ = sf.read(path)
            if data.ndim > 1:
                data = data[:, 0]
            self.noise_cache[path] = torch.from_numpy(data).float()
        return self.noise_cache[path]

    def __getitem__(self, index):
        try:
            return self._load(index)
        except (RuntimeError, OSError) as e:
            logger.warning(f"skip {index}: {e}")
            return self._load(random.randint(0, len(self) - 1))

    def _load(self, index):
        sample = self.data_list[index]
        clean_audio, sr = sf.read(sample['audio'])
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]
        clean = torch.from_numpy(clean_audio).float()
        seg = self.cfg['training_cfg']['segment_size']
        if self.split:
            if clean.size(0) >= seg:
                s = random.randint(0, clean.size(0) - seg)
                clean = clean[s:s + seg]
                start_sec = s / sr
            else:
                clean = F.pad(clean, (0, seg - clean.size(0)))
                start_sec = 0.0
            dur_sec = seg / sr
        else:
            start_sec, dur_sec = None, None
        video = load_video_frames(sample['video'], start_sec, dur_sec, self.face_size, self.fps)
        if self.split:
            video = pad_or_trim_video(video, max(1, int(dur_sec * self.fps)))
        if self.visual_augmentation:
            video = apply_visual_aug(video)
        if self.rir:
            clean = self.rir(clean)
        if self.noise_paths:
            snr_db = random.uniform(self.cfg['training_cfg']['snr_range'][0],
                                    self.cfg['training_cfg']['snr_range'][1])
            noisy = mix_audio(clean, self._get_noise(random.choice(self.noise_paths)), snr_db)
        else:
            noisy = clean.clone()
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        norm = torch.sqrt(noisy.size(1) / (torch.sum(noisy ** 2) + 1e-10))
        clean = clean * norm
        noisy = noisy * norm
        c_mag, c_pha, c_com = mag_phase_stft(clean, self.n_fft, self.hop, self.win, self.compress)
        n_mag, n_pha, _ = mag_phase_stft(noisy, self.n_fft, self.hop, self.win, self.compress)
        return (clean.squeeze(), c_mag.squeeze(), c_pha.squeeze(), c_com.squeeze(),
                n_mag.squeeze(), n_pha.squeeze(), video)

    def __len__(self):
        return len(self.data_list)

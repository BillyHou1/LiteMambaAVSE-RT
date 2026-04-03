#VCTK DEMAND audio only dataloader from SEMamba baseline.
import os
import json
import random
import torch
import torch.utils.data
import librosa
from models.stfts import mag_phase_stft
from models.pcs400 import cal_pcs

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class VCTKDemandDataset(torch.utils.data.Dataset):
    def __init__(self, clean_json, noisy_json, sampling_rate=16000,
                 segment_size=32000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=1.0, split=True, n_cache_reuse=1,
                 shuffle=True, device=None, pcs=False):
        self.noisy_paths = load_json(noisy_json)
        clean_paths = load_json(clean_json)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.noisy_paths)
        #map filename -> clean path so we can find the clean version of each noisy file
        self.clean_dict = {os.path.basename(p): p for p in clean_paths}
        self.sr = sampling_rate
        self.seg = segment_size
        self.n_fft = n_fft
        self.hop = hop_size
        self.win = win_size
        self.compress = compress_factor
        self.split = split
        self.n_cache_reuse = n_cache_reuse
        self.cached_clean = None
        self.cached_noisy = None
        self.cache_count = 0
        self.device = device
        self.pcs = pcs

    def __getitem__(self, index):
        if self.cache_count == 0:
            noisy_path = self.noisy_paths[index]
            clean_path = self.clean_dict.get(os.path.basename(noisy_path))
            noisy_audio, _ = librosa.load(noisy_path, sr=self.sr)
            clean_audio, _ = librosa.load(clean_path, sr=self.sr)
            if self.pcs:
                clean_audio = cal_pcs(clean_audio)
            self.cached_noisy = noisy_audio
            self.cached_clean = clean_audio
            self.cache_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean
            noisy_audio = self.cached_noisy
            self.cache_count -= 1

        clean = torch.FloatTensor(clean_audio)
        noisy = torch.FloatTensor(noisy_audio)
        norm = torch.sqrt(len(noisy) / torch.sum(noisy ** 2.0))
        clean = (clean * norm).unsqueeze(0)
        noisy = (noisy * norm).unsqueeze(0)

        if self.split:
            if clean.size(1) >= self.seg:
                s = random.randint(0, clean.size(1) - self.seg)
                clean = clean[:, s:s + self.seg]
                noisy = noisy[:, s:s + self.seg]
            else:
                clean = torch.nn.functional.pad(clean, (0, self.seg - clean.size(1)))
                noisy = torch.nn.functional.pad(noisy, (0, self.seg - noisy.size(1)))

        c_mag, c_pha, c_com = mag_phase_stft(clean, self.n_fft, self.hop, self.win, self.compress)
        n_mag, n_pha, _ = mag_phase_stft(noisy, self.n_fft, self.hop, self.win, self.compress)
        return (clean.squeeze(), c_mag.squeeze(), c_pha.squeeze(), c_com.squeeze(),
                n_mag.squeeze(), n_pha.squeeze())

    def __len__(self):
        return len(self.noisy_paths)

#RIR convolution
#Reference: 
#https://github.com/microsoft/DNS-Challenge
#https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
import json
import numpy as np
import torch
import torchaudio
from scipy.signal import fftconvolve

def load_rir(path, target_sr=16000):
    rir, sr = torchaudio.load(path)
    rir = rir[0].numpy()
    if sr != target_sr:
        rir = torchaudio.transforms.Resample(sr, target_sr)(
            torch.from_numpy(rir).float().unsqueeze(0)
        ).squeeze(0).numpy()
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-10)
    return rir.astype(np.float32)

def add_reverb(audio_np, rir_np):
    rev = fftconvolve(audio_np, rir_np, mode='full')[:len(audio_np)]
    orig_rms = np.sqrt(np.mean(audio_np ** 2) + 1e-10)
    rev_rms = np.sqrt(np.mean(rev ** 2) + 1e-10)
    rev = rev * (orig_rms / rev_rms)
    return rev.astype(np.float32)

class RIRAugmentor:
    def __init__(self, rir_json, prob=0.3, target_sr=16000):
        self.prob = prob
        self.target_sr = target_sr
        with open(rir_json, 'r') as f:
            self.rir_paths = json.load(f)
        self.cache = {}

    def _get_rir(self, idx):
        if idx not in self.cache:
            self.cache[idx] = load_rir(self.rir_paths[idx], self.target_sr)
        return self.cache[idx]

    def __call__(self, audio):
        if torch.rand(1).item() >= self.prob or len(self.rir_paths) == 0:
            return audio
        idx = torch.randint(len(self.rir_paths), (1,)).item()
        rir = self._get_rir(idx)
        is_tensor = isinstance(audio, torch.Tensor)
        audio_np = audio.numpy() if is_tensor else audio
        rev = add_reverb(audio_np, rir)
        if is_tensor:
            return torch.from_numpy(rev)
        return rev

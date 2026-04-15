#LRS2 AV dataloader
#Reference: 
#https://github.com/microsoft/MS-SNSD
#https://github.com/microsoft/DNS-Challenge
#https://github.com/ms-dot-k/AVSR
#https://github.com/nguyenvulebinh/AVSRCocktail
#https://github.com/RoyChao19477/SEMamba
import json
import random
import logging
import numpy as np
import torch
import av
import torchaudio
import torch.utils.data
import cv2

from models.stfts import mag_phase_stft
from dataloaders.augment_rir import RIRAugmentor
logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_audio(video_path, sr=16000):
    con = av.open(video_path)
    stream = con.streams.audio[0]
    orig_sr = stream.sample_rate
    frames = []
    for f in con.decode(stream):
        frames.append(f.to_ndarray())
    con.close()
    wav = np.concatenate(frames, axis=1)
    wav = wav.mean(0).astype('float32')
    wav = torch.from_numpy(wav)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        wav = resampler(wav.unsqueeze(0)).squeeze(0)
    return wav

def load_video_frames(video_path, start_sec=None, dur_sec=None, face_size=96, fps=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"can not open:{video_path}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * vfps) if start_sec is not None else 0
    n_frames = int(dur_sec * fps) if dur_sec is not None else total - start_frame
    n_frames = max(1, min(n_frames, total - start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        d = min(h, w)
        frame = frame[(h - d) // 2:(h + d) // 2, (w - d) // 2:(w + d) // 2]
        frame = cv2.resize(frame, (face_size, face_size))
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"no frames: {video_path}")
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    return video

def mix_audio(clean, noise, snr_db):
    n = clean.size(0)
    if noise.size(0) >= n:
        start = random.randint(0, noise.size(0) - n)
        noise = noise[start:start + n]
    else:
        times = n // noise.size(0) + 1
        noise = noise.repeat(times)
        noise = noise[:n]
    clean_pow = torch.mean(clean ** 2) + 1e-10
    noise_pow = torch.mean(noise ** 2) + 1e-10
    target_noise_pow = clean_pow / 10 ** (snr_db / 10.0)
    scale = torch.sqrt(target_noise_pow / noise_pow)
    return clean + scale * noise

def apply_visual_degradation(video, fps=25, min_dur_ms=200, max_dur_ms=500):
    Tv = video.shape[1]
    min_frames = max(3, int(min_dur_ms * fps / 1000))
    max_frames = max(min_frames + 1, int(max_dur_ms * fps / 1000))
    max_frames = min(max_frames, Tv - 2)
    if min_frames > max_frames:
        return video
    win_len = random.randint(min_frames, max_frames)
    start = random.randint(0, Tv - win_len - 1)
    t = torch.linspace(0, np.pi, win_len)
    curve = 0.5 * (1.0 + torch.cos(t))
    depth = random.uniform(0.3, 1.0)
    curve = 1.0 - depth * (1.0 - curve)
    curve = curve.reshape(1, win_len, 1, 1)
    deg_type = random.choice(['blur','blackout','freeze','lip_occlude','lighting'])
    if deg_type == 'blur':
        degraded = video[:, start:start + win_len].clone()
        for i in range(win_len):
            sigma = (1.0 - curve[0, i, 0, 0].item()) * 8.0 + 0.01
            if sigma > 0.5:
                k = int(sigma * 4)
                if k % 2 == 0:
                    k = k + 1
                if k < 3:
                    k = 3
                if k > 15:
                    k = 15
                for c in range(3):
                    f_np = degraded[c, i].numpy()
                    degraded[c, i] = torch.from_numpy(cv2.GaussianBlur(f_np, (k, k), sigma))
        video[:, start:start + win_len] = degraded
    elif deg_type == 'blackout':
        video[:, start:start + win_len] = video[:,start:start + win_len] * curve
    elif deg_type == 'freeze':
        frozen = video[:, start:start + 1].repeat(1, win_len, 1, 1)
        video[:, start:start + win_len] = curve * video[:, start:start + win_len] + (1.0 - curve) * frozen
    elif deg_type == 'lip_occlude':
        C, T, H, W = video.shape
        occ_top = int(H * random.uniform(0.45, 0.65))
        fill_val = random.uniform(0.2, 0.6)
        inv_curve = 1.0 - curve
        video[:, start:start + win_len, occ_top:, :] = curve * video[:, start:start + win_len, occ_top:, :] + inv_curve * fill_val
    elif deg_type == 'lighting':
        if random.random() < 0.5:
            dim_floor = random.uniform(0.1, 0.4)
            light_curve = curve * (1.0 - dim_floor) + dim_floor
            video[:, start:start + win_len] = video[:, start:start + win_len] * light_curve
        else:
            bright_ceil = random.uniform(0.7, 0.95)
            inv_curve = 1.0 - curve
            video[:, start:start + win_len] = video[:, start:start + win_len] * curve + bright_ceil * inv_curve
            video = torch.clamp(video, 0.0, 1.0)
    return video

class LRS2AVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json=None, sampling_rate=16000,
                 segment_size=32000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=1.0, snr_range=(-5, 20), face_size=96,
                 video_fps=25, split=True, shuffle=True,
                 rir_json=None, rir_prob=0.3,
                 visual_degradation_prob=0.0,
                 modality_conflict_prob=0.0,
                 cocktail_party_prob=0.0,
                 cocktail_num_speakers=1,
                 cocktail_mix_protocol='bernoulli'):
        self.entries = load_json(data_json)
        if noise_json:
            self.noise_paths = load_json(noise_json)
        else:
            self.noise_paths = []
        random.seed(1234)
        if shuffle:
            random.shuffle(self.entries)
        self.sr = sampling_rate
        self.seg = segment_size
        self.n_fft = n_fft
        self.hop = hop_size
        self.win = win_size
        self.compress = compress_factor
        self.snr_min = snr_range[0]
        self.snr_max = snr_range[1]
        self.face_size = face_size
        self.fps = video_fps
        self.split = split
        if rir_json:
            self.rir = RIRAugmentor(rir_json, rir_prob, sampling_rate)
        else:
            self.rir = None
        self.vis_deg_prob = visual_degradation_prob
        self.mod_conflict_prob = modality_conflict_prob
        self.cocktail_prob = cocktail_party_prob
        self.cocktail_num = cocktail_num_speakers
        self.cocktail_mix_protocol = cocktail_mix_protocol

    def _get_noise(self):
        path = random.choice(self.noise_paths)
        con = av.open(path)
        stream = con.streams.audio[0]
        sr = stream.sample_rate
        frames = []
        for f in con.decode(stream):
            frames.append(f.to_ndarray())
        con.close()
        wav = np.concatenate(frames, axis=1)
        wav = wav.mean(0).astype('float32')
        wav = torch.from_numpy(wav)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav.unsqueeze(0)).squeeze(0)
        return wav

    def __getitem__(self, index):
        try:
            return self._load(index)
        except (RuntimeError, OSError) as e:
            logger.warning(f"skip {index}: {e}")
            return self._load(random.randint(0, len(self) - 1))

    def _load(self, index):
        entry = self.entries[index]
        if isinstance(entry, dict):
            vpath = entry['video']
        else:
            vpath = entry
        clean = extract_audio(vpath, self.sr)
        vis_corrupted = False

        if self.split:
            if clean.size(0) >= self.seg:
                s = random.randint(0, clean.size(0) - self.seg)
                clean = clean[s:s + self.seg]
                start_sec = s / self.sr
            else:
                pad_len = self.seg - clean.size(0)
                clean = torch.nn.functional.pad(clean, (0, pad_len))
                start_sec = 0.0
            dur_sec = self.seg / self.sr
        else:
            start_sec = None
            dur_sec = None

        video = load_video_frames(vpath, start_sec, dur_sec, self.face_size, self.fps)

        if self.split:
            expected_v = int(self.seg * self.fps / self.sr)
            Tv = video.shape[1]
            if Tv < expected_v:
                pad = video[:, Tv - 1:Tv, :, :].repeat(1, expected_v - Tv, 1, 1)
                video = torch.cat([video, pad], dim=1)
            elif Tv > expected_v:
                video = video[:, :expected_v, :, :]
        if self.vis_deg_prob > 0 and self.split and random.random() <= self.vis_deg_prob:
            video = apply_visual_degradation(video, fps=self.fps, min_dur_ms=200, max_dur_ms=500)
            vis_corrupted = True
        if self.mod_conflict_prob > 0 and self.split and random.random() < self.mod_conflict_prob:
            swap_idx = random.randint(0, len(self.entries) - 1)
            swap_entry = self.entries[swap_idx]
            if isinstance(swap_entry, dict):
                swap_vpath = swap_entry['video']
            else:
                swap_vpath = swap_entry
            try:
                video = load_video_frames(swap_vpath, start_sec, dur_sec, self.face_size, self.fps)
                Tv = video.shape[1]
                if self.split:
                    if Tv < expected_v:
                        pad = video[:, Tv - 1:Tv, :, :].repeat(1, expected_v - Tv, 1, 1)
                        video = torch.cat([video, pad], dim=1)
                    elif Tv > expected_v:
                        video = video[:, :expected_v, :, :]
                vis_corrupted = True
            except (RuntimeError, OSError) as e:
                logger.warning(f"modality conflict failed: {e}")

        if self.rir:
            clean = self.rir(clean)

        if self.cocktail_mix_protocol == 'raven' and self.cocktail_prob > 0:
            r = random.random()
            if r < 0.5:
                n_spk = 1
                add_env_noise = True
            elif r < 0.75:
                n_spk = 2
                add_env_noise = False
            else:
                n_spk = 1
                add_env_noise = False
            do_cocktail = True
        else:
            n_spk = self.cocktail_num if hasattr(self, 'cocktail_num') else 1
            add_env_noise = bool(self.noise_paths)
            do_cocktail = self.cocktail_prob > 0 and random.random() < self.cocktail_prob

        if add_env_noise and self.noise_paths:
            snr = random.uniform(self.snr_min, self.snr_max)
            noisy = mix_audio(clean, self._get_noise(), snr)
        else:
            noisy = clean.clone()

        if do_cocktail:
            interfering_sum = torch.zeros_like(clean)
            for _ in range(n_spk):
                int_idx = random.randint(0, len(self.entries) - 1)
                int_entry = self.entries[int_idx]
                if isinstance(int_entry, dict):
                    int_vpath = int_entry['video']
                else:
                    int_vpath = int_entry
                try:
                    int_audio = extract_audio(int_vpath, self.sr)
                    if int_audio.shape[0] > clean.shape[0]:
                        i_start = random.randint(0, int_audio.shape[0] - clean.shape[0])
                        int_audio = int_audio[i_start:i_start + clean.shape[0]]
                    elif int_audio.shape[0] < clean.shape[0]:
                        pad_len = clean.shape[0] - int_audio.shape[0]
                        int_audio = torch.nn.functional.pad(int_audio, (0, pad_len))
                    interfering_sum = interfering_sum + int_audio
                except (RuntimeError, OSError) as e:
                    logger.warning(f"cocktail mix failed: {e}")
            if interfering_sum.abs().max() > 0:
                snr_db = random.uniform(self.snr_min, self.snr_max)
                clean_pow = torch.mean(clean ** 2) + 1e-10
                int_pow = torch.mean(interfering_sum ** 2) + 1e-10
                target_pow = clean_pow / 10 ** (snr_db / 10.0)
                scale = torch.sqrt(target_pow / int_pow)
                noisy = noisy + scale * interfering_sum

        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        energy = torch.sum(noisy ** 2) + 1e-10
        norm = torch.sqrt(noisy.size(1) / energy)
        clean = clean * norm
        noisy = noisy * norm
        c_mag, c_pha, c_com = mag_phase_stft(clean, self.n_fft, self.hop, self.win, self.compress)
        n_mag, n_pha, _ = mag_phase_stft(noisy, self.n_fft, self.hop, self.win, self.compress)

        return (clean.squeeze(), c_mag.squeeze(), c_pha.squeeze(), c_com.squeeze(),
                n_mag.squeeze(), n_pha.squeeze(), video,
                torch.tensor(vis_corrupted, dtype=torch.float32))

    def __len__(self):
        return len(self.entries)

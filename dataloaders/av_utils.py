import json
import os
import random
import numpy as np
import torch
import cv2
from models.stfts import mag_phase_stft

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_video_frames(video_path, start_sec=None, dur_sec=None, face_size=96, fps=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
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
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0

def extract_audio_from_video(video_path, target_sr=16000):
    import av
    container = av.open(video_path)
    stream = container.streams.audio[0]
    frames = [f.to_ndarray() for f in container.decode(stream)]
    orig_sr = stream.sample_rate
    container.close()
    wav = torch.from_numpy(np.concatenate(frames, axis=1).mean(0).astype('float32'))
    if orig_sr != target_sr:
        import torchaudio
        wav = torchaudio.transforms.Resample(orig_sr, target_sr)(wav.unsqueeze(0)).squeeze(0)
    return wav

def mix_audio(clean, noise, snr_db):
    n = clean.size(0)
    if noise.size(0) >= n:
        s = random.randint(0, noise.size(0) - n)
        noise = noise[s:s + n]
    else:
        noise = noise.repeat(n // noise.size(0) + 1)[:n]
    clean_pow = torch.mean(clean ** 2) + 1e-10
    noise_pow = torch.mean(noise ** 2) + 1e-10
    scale = torch.sqrt(clean_pow / (10 ** (snr_db / 10.0) * noise_pow))
    return clean + scale * noise

def apply_visual_aug(frames):
    r = random.random()
    if r < 0.08:
        return torch.zeros_like(frames)
    elif r < 0.18:
        t_len = frames.size(1)
        mask = torch.rand(t_len) > random.uniform(0.2, 0.8)
        frames[:, ~mask] = 0
        return frames
    elif r < 0.28:
        return torch.clamp(frames + torch.randn_like(frames) * random.uniform(0.1, 0.6), 0, 1)
    elif r < 0.35:
        c, t_len, h, w = frames.shape
        k = random.choice([3, 5, 7])
        return torch.nn.functional.avg_pool2d(
            frames.reshape(c * t_len, 1, h, w), k, stride=1, padding=k // 2
        ).reshape(c, t_len, h, w)
    elif r < 0.40:
        return frames * random.uniform(0.1, 0.5)
    return frames

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
    deg_type = random.choice(['blur', 'blackout', 'freeze', 'lip_occlude', 'lighting'])
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
        video[:, start:start + win_len] = video[:, start:start + win_len] * curve
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

def pad_or_trim_video(frames, target_t):
    t_len = frames.shape[1]
    if t_len < target_t:
        pad = frames[:, -1:].repeat(1, target_t - t_len, 1, 1)
        return torch.cat([frames, pad], dim=1)
    elif t_len > target_t:
        return frames[:, :target_t]
    return frames

def normalize_and_stft(clean, noisy, n_fft, hop, win, compress):
    clean = clean.unsqueeze(0)
    noisy = noisy.unsqueeze(0)
    norm = torch.sqrt(noisy.size(1) / (torch.sum(noisy ** 2) + 1e-10))
    clean = clean * norm
    noisy = noisy * norm
    c_mag, c_pha, c_com = mag_phase_stft(clean, n_fft, hop, win, compress)
    n_mag, n_pha, _ = mag_phase_stft(noisy, n_fft, hop, win, compress)
    return (clean.squeeze(), c_mag.squeeze(), c_pha.squeeze(), c_com.squeeze(),
            n_mag.squeeze(), n_pha.squeeze())

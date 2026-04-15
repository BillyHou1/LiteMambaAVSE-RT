#LiteAVSEMamba inference
import os
import argparse
import torch
import librosa
import soundfile as sf
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import LiteAVSEMamba
from dataloaders.av_utils import load_video_frames
from utils.util import load_config, load_checkpoint

def enhance_file(model, audio_path, video_path, cfg, device):
    n_fft = cfg['stft_cfg']['n_fft']
    hop = cfg['stft_cfg']['hop_size']
    win = cfg['stft_cfg']['win_size']
    compress = cfg['model_cfg']['compress_factor']
    sr = cfg['stft_cfg']['sampling_rate']
    face_size = cfg.get('visual_cfg', {}).get('face_size', 96)
    fps = cfg.get('visual_cfg', {}).get('video_fps', 25)

    noisy, _ = librosa.load(audio_path, sr=sr)
    noisy = torch.FloatTensor(noisy).to(device)
    norm = torch.sqrt(len(noisy) / (torch.sum(noisy ** 2) + 1e-10))
    noisy = (noisy * norm).unsqueeze(0)

    n_mag, n_pha, _ = mag_phase_stft(noisy, n_fft, hop, win, compress)

    video = None
    if video_path is not None and os.path.exists(video_path):
        video = load_video_frames(video_path, face_size=face_size, fps=fps)
        video = video.unsqueeze(0).to(device)

    with torch.no_grad():
        mag_g, pha_g, _ = model(n_mag, n_pha, video)

    out = mag_phase_istft(mag_g, pha_g, n_fft, hop, win, compress)
    out = out / norm
    return out.squeeze().cpu().numpy(), sr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None)
    parser.add_argument('--input_audio', default=None)
    parser.add_argument('--input_video', default=None)
    parser.add_argument('--output_folder', default='results_av')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    if args.input_folder is None and (args.input_audio is None or args.input_video is None):
        raise ValueError("need --input_folder or both --input_audio and --input_video")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    model = LiteAVSEMamba(cfg).to(device)
    state = load_checkpoint(args.checkpoint, device)
    if 'generator' in state:
        state = state['generator']
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs(args.output_folder, exist_ok=True)

    if args.input_audio is not None and args.input_video is not None:
        out, sr = enhance_file(model, args.input_audio, args.input_video, cfg, device)
        fname = os.path.basename(args.input_audio)
        out_path = os.path.join(args.output_folder, fname)
        sf.write(out_path, out, sr, 'PCM_16')
        print(f"saved: {out_path}")

    elif args.input_folder is not None:
        for fname in sorted(os.listdir(args.input_folder)):
            if not fname.endswith('.wav'):
                continue
            base = os.path.splitext(fname)[0]
            audio_path = os.path.join(args.input_folder, fname)
            video_path = os.path.join(args.input_folder, base + '.mp4')
            if not os.path.exists(video_path):
                video_path = None
            print(f"processing: {fname}")
            out, sr = enhance_file(model, audio_path, video_path, cfg, device)
            out_path = os.path.join(args.output_folder, fname)
            sf.write(out_path, out, sr, 'PCM_16')
            print(f"saved: {out_path}")

if __name__ == '__main__':
    main()

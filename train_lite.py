import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import LiteAVSEMamba, SEMamba
from models.discriminator import MetricDiscriminator, batch_pesq
from models.loss import pesq_score, phase_losses, si_sdr_loss, si_sdr_score, stoi_score
from utils.util import (
    save_checkpoint, scan_checkpoint, load_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
    setup_cache_dirs, check_loss_health, safe_backward, safe_step,
    save_best_model, save_checkpoint_with_meta, load_best_pesq_from_ckpt,
)

torch.backends.cudnn.benchmark = True


def setup_optimizer(generator, cfg):
    lr = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])
    use_diff_lr = cfg['training_cfg'].get('use_differential_lr', False)
    if use_diff_lr:
        audio_modules = ['dense_encoder', 'ts_mamba', 'mask_decoder', 'phase_decoder']
        vis_modules = ['visual_encoder']
        audio_scale = cfg['training_cfg'].get('audio_lr_scale', 1.0)
        vis_scale = cfg['training_cfg'].get('visual_lr_scale', 0.2)
        audio_params, vis_params, fusion_params = [], [], []
        for name, param in generator.named_parameters():
            if param.requires_grad:
                if any(name.startswith(m) or name.startswith(f'module.{m}') for m in audio_modules):
                    audio_params.append(param)
                elif any(name.startswith(m) or name.startswith(f'module.{m}') for m in vis_modules):
                    vis_params.append(param)
                else:
                    fusion_params.append(param)
        param_groups = [
            {'params': audio_params, 'lr': lr * audio_scale},
            {'params': vis_params, 'lr': lr * vis_scale},
            {'params': fusion_params, 'lr': lr},
        ]
        return optim.AdamW(param_groups, betas=betas)
    else:
        return optim.AdamW(
            filter(lambda p: p.requires_grad, generator.parameters()),
            lr=lr, betas=betas)

def setup_scheduler(optim_g, cfg, last_epoch):
    lr_decay = cfg['training_cfg']['lr_decay']
    return optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)

def create_dataset(cfg, train=True, split=True):
    dataset_type = cfg['data_cfg'].get('dataset_type', 'grid')
    if train:
        data_json = cfg['data_cfg']['train_data_json']
        noise_json = cfg['data_cfg'].get('train_noise_json', None)
    else:
        data_json = cfg['data_cfg']['valid_data_json']
        noise_json = cfg['data_cfg'].get('valid_noise_json', None)
    vis_cfg = cfg.get('visual_cfg', {})
    snr_range = cfg['training_cfg'].get('snr_range', [-5, 20])
    rir_json = cfg['data_cfg'].get('rir_json', None) if train else None
    rir_prob = cfg['data_cfg'].get('rir_prob', 0.3)
    common = dict(
        data_json=data_json, noise_json=noise_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        snr_range=tuple(snr_range),
        face_size=vis_cfg.get('face_size', 96),
        video_fps=vis_cfg.get('video_fps', 25),
        split=split, shuffle=train,
        rir_json=rir_json, rir_prob=rir_prob)

    if dataset_type == 'grid':
        from dataloaders.dataloader_grid import GRIDAVDataset
        vis_aug = train and cfg.get('training_cfg', {}).get('visual_augmentation', False)
        return GRIDAVDataset(visual_augmentation=vis_aug, **common)
    elif dataset_type == 'lrs2':
        from dataloaders.dataloader_lrs import LRS2AVDataset
        vis_deg = cfg['training_cfg'].get('visual_degradation_prob', 0.0) if train else 0.0
        mod_conf = cfg['training_cfg'].get('modality_conflict_prob', 0.0) if train else 0.0
        cocktail = cfg['training_cfg'].get('cocktail_party_prob', 0.0) if train else 0.0
        cocktail_num = cfg['training_cfg'].get('cocktail_num_speakers', 1)
        cocktail_protocol = cfg['training_cfg'].get('cocktail_mix_protocol', 'bernoulli')
        return LRS2AVDataset(visual_degradation_prob=vis_deg,
                            modality_conflict_prob=mod_conf,
                            cocktail_party_prob=cocktail,
                            cocktail_num_speakers=cocktail_num,
                            cocktail_mix_protocol=cocktail_protocol, **common)
    elif dataset_type == 'vox':
        from dataloaders.dataloader_vox import VoxCelebAVDataset
        min_len = cfg['data_cfg'].get('min_audio_len', 8000)
        return VoxCelebAVDataset(min_audio_len=min_len, **common)
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

def create_dataloader(dataset, cfg, train=True):
    batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1
    return DataLoader(dataset, num_workers=num_workers, shuffle=train,
                      batch_size=batch_size, pin_memory=True, drop_last=train)

def load_ckpts(exp_path, device):
    if os.path.isdir(exp_path):
        cp_g = scan_checkpoint(exp_path, 'g_')
        cp_do = scan_checkpoint(exp_path, 'do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_g = load_checkpoint(cp_g, device)
        state_do = load_checkpoint(cp_do, device)
        return state_g, state_do, state_do['steps'] + 1, state_do['epoch']
    return None, None, 0, -1

def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress = cfg['model_cfg']['compress_factor']

    if num_gpus > 1:
        initialize_process_group(cfg, rank)
    device = torch.device(f'cuda:{rank}')

    model_type = cfg['model_cfg'].get('model_type', 'liteavse')
    if model_type == 'semamba':
        generator = SEMamba(cfg).to(device)
    else:
        generator = LiteAVSEMamba(cfg).to(device)
    use_gan = cfg['training_cfg'].get('use_gan', True)
    if use_gan:
        discriminator = MetricDiscriminator().to(device)
    if rank == 0:
        log_model_info(rank, generator, args.exp_path)
        total = sum(p.numel() for p in generator.parameters())
        trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"Total params: {total:,}, Trainable: {trainable:,}")
        if use_gan:
            print(f"GAN on, Disc params: {sum(p.numel() for p in discriminator.parameters()):,}")

    state_g, state_do, steps, last_epoch = load_ckpts(args.exp_path, device)
    if state_g is not None:
        generator.load_state_dict(state_g['generator'], strict=False)
        if use_gan and state_do is not None and 'discriminator' in state_do:
            discriminator.load_state_dict(state_do['discriminator'], strict=False)

    if num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        if use_gan:
            discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = setup_optimizer(generator, cfg)
    if state_do is not None:
        optim_g.load_state_dict(state_do['optim_g'])
    if use_gan:
        lr = cfg['training_cfg']['learning_rate']
        betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])
        optim_d = optim.AdamW(discriminator.parameters(), lr=lr, betas=betas)
        if state_do is not None and 'optim_d' in state_do:
            optim_d.load_state_dict(state_do['optim_d'])
    sched_g = setup_scheduler(optim_g, cfg, last_epoch)
    if use_gan:
        sched_d = optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=cfg['training_cfg']['lr_decay'], last_epoch=last_epoch)

    trainset = create_dataset(cfg, train=True, split=True)
    train_loader = create_dataloader(trainset, cfg, train=True)
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=False)
        val_loader = create_dataloader(validset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    generator.train()
    if use_gan:
        discriminator.train()
    best_pesq, best_pesq_step = load_best_pesq_from_ckpt(state_do)
    nan_count = 0

    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            t_start = time.time()
            print(f"Epoch: {epoch + 1}")

        for i, batch in enumerate(train_loader):
            if rank == 0:
                t_batch = time.time()

            if len(batch) == 8:
                c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video, vis_deg = batch # [B,1,F,T] spec, [B,3,Tv,H,W] video
                vis_deg = vis_deg.to(device, non_blocking=True)
            else:
                c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video = batch
                vis_deg = None

            c_aud = c_aud.to(device, non_blocking=True)
            c_mag = c_mag.to(device, non_blocking=True)
            c_pha = c_pha.to(device, non_blocking=True)
            c_com = c_com.to(device, non_blocking=True)
            n_mag = n_mag.to(device, non_blocking=True)
            n_pha = n_pha.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)

            if model_type == 'semamba':
                mag_g, pha_g, com_g = generator(n_mag, n_pha)
                intermediates = {}
            else:
                mag_g, pha_g, com_g, intermediates = generator(n_mag, n_pha, video,
                    return_intermediates=True, visual_degraded=vis_deg)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress)

            if use_gan:
                bs = c_mag.shape[0]
                one_labels = torch.ones(bs, device=device)
                audio_list_r = list(c_aud.cpu().numpy())
                audio_list_g = list(audio_g.detach().cpu().numpy())
                bp_score = batch_pesq(audio_list_r, audio_list_g, cfg)

                optim_d.zero_grad()
                metric_r = discriminator(c_mag, c_mag)
                metric_g = discriminator(c_mag, mag_g.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
                if bp_score is not None:
                    loss_disc_g = F.mse_loss(bp_score.to(device), metric_g.flatten())
                else:
                    loss_disc_g = 0
                loss_disc = loss_disc_r + loss_disc_g
                loss_disc.backward()
                optim_d.step()

            optim_g.zero_grad()
            loss_mag = F.mse_loss(c_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(c_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(c_com, com_g) * 2
            loss_time = F.l1_loss(c_aud, audio_g)
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_cfg = cfg['training_cfg']['loss']
            si_sdr_w = loss_cfg.get('si_sdr', 0.0)
            loss_si_sdr = si_sdr_loss(c_aud, audio_g) if si_sdr_w > 0 else torch.tensor(0.0, device=device)

            if use_gan:
                metric_g_gen = discriminator(c_mag, mag_g)
                loss_metric = F.mse_loss(metric_g_gen.flatten(), one_labels)
                metric_w = loss_cfg.get('metric', 0.05)
            else:
                loss_metric = torch.tensor(0.0, device=device)
                metric_w = 0.0

            loss_all = (
                loss_mag * loss_cfg['magnitude'] +
                loss_pha * loss_cfg['phase'] +
                loss_com * loss_cfg['complex'] +
                loss_time * loss_cfg['time'] +
                loss_con * loss_cfg['consistancy'] +
                loss_si_sdr * si_sdr_w +
                loss_metric * metric_w)

            aux_w = loss_cfg.get('aux_visual', 0.0)
            if aux_w > 0:
                loss_aux = intermediates.get('aux_loss', torch.tensor(0.0, device=device))
                loss_all = loss_all + loss_aux * aux_w

            alpha_smooth_w = loss_cfg.get('alpha_smooth', 0.0)
            if alpha_smooth_w > 0 and 'alpha' in intermediates:
                alpha = intermediates['alpha'] # [B,T,1]
                alpha_diff = (alpha[:, 1:, :] - alpha[:, :-1, :]).abs()
                loss_alpha_smooth = alpha_diff.mean()
                sat_thresh = loss_cfg.get('alpha_sat_threshold', 0.4)
                alpha_sat = ((alpha - 0.5).abs() - sat_thresh).clamp(min=0).mean()
                loss_all = loss_all + (loss_alpha_smooth + alpha_sat * 0.5) * alpha_smooth_w

            alpha_ent_w = loss_cfg.get('alpha_entropy', 0.0)
            if alpha_ent_w > 0 and 'alpha' in intermediates:
                a = intermediates['alpha'].clamp(1e-6, 1 - 1e-6)
                entropy = -(a * a.log() + (1 - a) * (1 - a).log()).mean()
                loss_all = loss_all - entropy * alpha_ent_w

            gate_w = loss_cfg.get('gate_supervision', 0.0)
            if gate_w > 0 and 'gate_loss' in intermediates:
                loss_all = loss_all + intermediates['gate_loss'] * gate_w

            healthy, nan_count, should_reload = check_loss_health(loss_all, nan_count)
            if should_reload:
                sys.exit(1)
            if not healthy:
                steps += 1
                continue

            gen_params = generator.module.parameters() if num_gpus > 1 else generator.parameters()
            if not safe_backward(loss_all, optim_g, max_grad_norm=1.0, model_params=gen_params):
                steps += 1
                continue
            if not safe_step(optim_g):
                steps += 1
                continue

            gen_model = generator.module if num_gpus > 1 else generator
            if hasattr(gen_model, 'current_step'):
                gen_model.current_step = steps

            if rank == 0:
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    with torch.no_grad():
                        mag_err = F.mse_loss(c_mag, mag_g).item()
                        pha_err = (loss_ip + loss_gd + loss_iaf).item()
                        com_err = F.mse_loss(c_com, com_g).item()
                        time_err = F.l1_loss(c_aud, audio_g).item()
                        con_err = F.mse_loss(com_g, rec_com).item()
                        si_sdr_val = loss_si_sdr.item() if si_sdr_w > 0 else 0.0

                    gate_info = ''
                    if 'alpha' in intermediates:
                        a = intermediates['alpha']
                        gate_info += f', a={a.mean().item():.3f}+/-{a.std().item():.3f}'
                    if 'freq_gate' in intermediates:
                        g = intermediates['freq_gate']
                        gate_info += f', fg={g.mean().item():.3f}+/-{g.std().item():.3f}'
                    if 'gate_loss' in intermediates:
                        gate_info += f', gsup={intermediates["gate_loss"].item():.4f}'

                    disc_info = f', D: {loss_disc.item():4.3f}' if use_gan else ''
                    print(f'Steps: {steps:d}, Loss: {loss_all.item():4.3f}, '
                          f'Mag: {mag_err:4.3f}, Pha: {pha_err:4.3f}, Com: {com_err:4.3f}, '
                          f'Time: {time_err:4.3f}, Con: {con_err:4.3f}, SiSDR: {si_sdr_val:4.3f}, '
                          f's/b: {time.time() - t_batch:4.3f}{disc_info}{gate_info}')

                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    gen = generator.module if num_gpus > 1 else generator
                    save_checkpoint(
                        f"{args.exp_path}/g_{steps:08d}.pth",
                        {'generator': gen.state_dict()})
                    do_state = {'optim_g': optim_g.state_dict(), 'steps': steps, 'epoch': epoch}
                    if use_gan:
                        disc = discriminator.module if num_gpus > 1 else discriminator
                        do_state['discriminator'] = disc.state_dict()
                        do_state['optim_d'] = optim_d.state_dict()
                    save_checkpoint_with_meta(
                        f"{args.exp_path}/do_{steps:08d}.pth",
                        do_state, best_pesq, best_pesq_step)

                if steps % cfg['env_setting']['summary_interval'] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_all.item(), steps)
                    sw.add_scalar("Training/Magnitude Loss", F.mse_loss(c_mag, mag_g).item(), steps)
                    sw.add_scalar("Training/Phase Loss", (loss_ip + loss_gd + loss_iaf).item(), steps)
                    sw.add_scalar("Training/Time Loss", F.l1_loss(c_aud, audio_g).item(), steps)
                    if use_gan:
                        sw.add_scalar("Training/Discriminator Loss", loss_disc.item(), steps)
                        sw.add_scalar("Training/Metric Loss", loss_metric.item(), steps)
                    if 'alpha' in intermediates:
                        sw.add_scalar("Gates/alpha_mean", intermediates['alpha'].mean().item(), steps)
                        sw.add_scalar("Gates/alpha_std", intermediates['alpha'].std().item(), steps)
                    if 'freq_gate' in intermediates:
                        sw.add_scalar("Gates/fsvg_mean", intermediates['freq_gate'].mean().item(), steps)
                    if 'gate_loss' in intermediates:
                        sw.add_scalar("Gates/gate_supervision", intermediates['gate_loss'].item(), steps)

                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_tot = 0
                    n_val = cfg['env_setting'].get('validation_samples', 500)
                    with torch.no_grad():
                        for j, batch in enumerate(val_loader):
                            if j >= n_val:
                                break
                            if len(batch) == 8:
                                c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video, _ = batch
                            else:
                                c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video = batch
                            c_aud = c_aud.to(device)
                            c_mag = c_mag.to(device)
                            n_mag = n_mag.to(device)
                            n_pha = n_pha.to(device)
                            video = video.to(device)
                            t_cfg = cfg['training_cfg']
                            s_cfg = cfg['stft_cfg']
                            max_t = t_cfg['segment_size'] // s_cfg['hop_size'] + 1
                            vis_cfg = cfg.get('visual_cfg', {})
                            max_v = int(t_cfg['segment_size'] * vis_cfg.get('video_fps', 25) / s_cfg['sampling_rate'])
                            if n_mag.shape[2] > max_t:
                                n_mag = n_mag[:, :, :max_t]
                                n_pha = n_pha[:, :, :max_t]
                                c_mag = c_mag[:, :, :max_t]
                                c_aud = c_aud[:, :t_cfg['segment_size']]
                                video = video[:, :, :max_v, :, :]

                            if model_type == 'semamba':
                                mag_g, pha_g, com_g = generator(n_mag, n_pha)
                            else:
                                mag_g, pha_g, com_g = generator(n_mag, n_pha, video)
                            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress)
                            audios_r += torch.split(c_aud, 1, dim=0) # [1,T] * B
                            audios_g += torch.split(audio_g, 1, dim=0)
                            val_mag_tot += F.mse_loss(c_mag, mag_g).item()

                        n_ran = len(audios_r)
                        val_mag_err = val_mag_tot / max(n_ran, 1)
                        val_pesq = pesq_score(audios_r, audios_g, cfg)
                        if isinstance(val_pesq, torch.Tensor):
                            val_pesq = val_pesq.item()
                        val_stoi = stoi_score(audios_r, audios_g, cfg)
                        val_sisdr = si_sdr_score(audios_r, audios_g)

                        print(f'Steps: {steps:d}, PESQ: {val_pesq:4.3f}, STOI: {val_stoi:4.3f}, '
                              f'SI-SDR: {val_sisdr:4.2f}dB, Mag: {val_mag_err:4.3f}')
                        sw.add_scalar("Validation/PESQ", val_pesq, steps)
                        sw.add_scalar("Validation/STOI", val_stoi, steps)
                        sw.add_scalar("Validation/SI-SDR", val_sisdr, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)

                    generator.train()
                    if val_pesq >= best_pesq:
                        best_pesq = val_pesq
                        best_pesq_step = steps
                        save_best_model(args.exp_path, generator, num_gpus, best_pesq, best_pesq_step)
                    print(f"Best PESQ: {best_pesq} at step {best_pesq_step}")

            steps += 1

            max_steps = cfg['training_cfg'].get('max_steps', None)
            if max_steps is not None and steps >= max_steps:
                if rank == 0:
                    print(f"Hit max_steps={max_steps}, stopping training.")
                return

        sched_g.step()
        if use_gan:
            sched_d.step()
        if rank == 0:
            print(f'Time for epoch {epoch + 1}: {int(time.time() - t_start)} sec\n')

def main():
    setup_cache_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='LiteAVSE')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    avail_gpus = torch.cuda.device_count()
    if num_gpus > avail_gpus:
        warnings.warn(f"Config num_gpus={num_gpus} > available={avail_gpus}. Using {avail_gpus}.")
        cfg['env_setting']['num_gpus'] = avail_gpus
        num_gpus = avail_gpus
    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        print_gpu_info(torch.cuda.device_count(), cfg)
    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)


if __name__ == '__main__':
    main()

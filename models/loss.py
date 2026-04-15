# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py
import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed

def phase_losses(phase_r, phase_g, cfg):
    """
    Calculate phase losses including in-phase loss, gradient delay loss, 
    and integrated absolute frequency loss between reference and generated phases.
    
    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        h (object): Configuration object containing parameters like n_fft.
    
    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension
    
    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    
    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    
    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    
    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    
    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    
    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    """
    Anti-wrapping function to adjust phase values within the range of -pi to pi.
    
    Args:
        x (torch.Tensor): Input tensor representing phase differences.
    
    Returns:
        torch.Tensor: Adjusted tensor with phase values wrapped within -pi to pi.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def pesq_score(utts_r, utts_g, cfg):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score for pairs of reference and generated utterances.
    
    Args:
        utts_r (list of torch.Tensor): List of reference utterances.
        utts_g (list of torch.Tensor): List of generated utterances.
        h (object): Configuration object containing parameters like sampling_rate.
    
    Returns:
        float: Mean PESQ score across all pairs of utterances.
    """
    def eval_pesq(clean_utt, esti_utt, sr):
        """
        Evaluate PESQ score for a single pair of clean and estimated utterances.
        
        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.
        
        Returns:
            float: PESQ score or -1 in case of an error.
        """
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt)
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1
        return pesq_score
    
    # Parallel processing of PESQ score computation
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))
    
    # Calculate mean PESQ score
    pesq_score = np.mean(pesq_scores)
    return pesq_score


def si_sdr_loss(reference, estimation):
    min_len = min(reference.shape[-1], estimation.shape[-1])
    reference = reference[..., :min_len]
    estimation = estimation[..., :min_len]
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimation = estimation - estimation.mean(dim=-1, keepdim=True)
    dot = torch.sum(reference * estimation, dim=-1, keepdim=True)
    s_ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + 1e-8
    proj = dot * reference / s_ref_energy
    noise = estimation - proj
    proj_energy = torch.sum(proj ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1) + 1e-8
    si_sdr = 10 * torch.log10(proj_energy / noise_energy + 1e-8)
    return -si_sdr.mean()

def si_sdr_score(utts_r, utts_g):
    scores = []
    for ref, est in zip(utts_r, utts_g):
        ref = ref.squeeze().cpu().numpy()
        est = est.squeeze().cpu().numpy()
        ref = ref.astype(np.float64)
        est = est.astype(np.float64)
        min_len = min(len(ref), len(est))
        ref, est = ref[:min_len], est[:min_len]
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        dot = np.sum(ref * est)
        s_ref_energy = np.sum(ref ** 2) + 1e-8
        proj = dot * ref / s_ref_energy
        noise = est - proj
        proj_energy = np.sum(proj ** 2)
        noise_energy = np.sum(noise ** 2) + 1e-8
        val = 10 * np.log10(proj_energy / noise_energy + 1e-8)
        scores.append(val)
    return np.mean(scores)

def stoi_score(utts_r, utts_g, cfg):
    from pystoi import stoi
    sr = cfg['stft_cfg']['sampling_rate']
    def eval_stoi(clean_utt, esti_utt, sr):
        try:
            min_len = min(len(clean_utt), len(esti_utt))
            return stoi(clean_utt[:min_len], esti_utt[:min_len], sr, extended=False)
        except Exception as e:
            print(f"Error computing STOI: {e}")
            return -1
    stoi_scores = Parallel(n_jobs=30)(delayed(eval_stoi)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        sr
    ) for i in range(len(utts_r)))
    return np.mean(stoi_scores)

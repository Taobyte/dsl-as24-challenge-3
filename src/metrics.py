import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
import einops

from obspy.signal.cross_correlation import correlate
from obspy.signal.trigger import z_detect

# ======================================= NUMPY METRICS ===================================================


def cross_correlation(eq: ndarray, denoised_eq: ndarray, shift: int = 0) -> float:
    max_eq = np.max(eq) + 1e-12
    max_denoised = np.max(denoised_eq) + 1e-12
    corr = correlate(eq / max_eq, denoised_eq / max_denoised, shift=shift)
    return np.max(corr)


def max_amplitude_difference(eq: ndarray, denoised_eq: ndarray) -> float:
    max_eq = np.max(eq) + 1e-12
    max_denoised = np.max(denoised_eq)
    return np.abs(max_denoised / max_eq)


def find_onset(denoised_eq, threshold=0.05, nsta=20):
    zfunc = z_detect(denoised_eq, nsta=nsta)
    zfunc -= np.mean(zfunc[:500])
    return np.argmax(zfunc[700:] > threshold) + 700


def p_wave_onset_difference(eq: ndarray, denoised_eq: ndarray, shift: int) -> float:
    ground_truth = 6000 - shift
    denoised_p_wave_onset = find_onset(denoised_eq)

    return np.abs(ground_truth - denoised_p_wave_onset)


# ======================================= TORCH METRICS ==================================================


def cross_correlation_torch(
    eq: torch.Tensor, denoised_eq: torch.Tensor
) -> torch.Tensor:
    """
    Pytorch's Conv1d is implemented as cross correlation
    """

    B, C, T = eq.shape
    eq -= eq.mean(dim=-1, keepdim=True)
    denoised_eq -= denoised_eq.mean(dim=-1, keepdim=True)
    eq_reshaped = einops.rearrange(eq, "b c t -> (b c) t")
    denoised_eq_reshaped = einops.rearrange(denoised_eq, "b c t -> (b c) t")

    result = F.conv1d(
        eq_reshaped, denoised_eq_reshaped.unsqueeze(1), padding=T - 1, groups=B * C
    )
    result = einops.rearrange(result, "(b channels) t -> b channels t", channels=C)

    # normalization
    norm = (
        torch.sum(eq**2, dim=-1, keepdim=True)
        * torch.sum(denoised_eq**2, dim=-1, keepdim=True)
    ) ** 0.5
    result = result / (norm + 1e-12)
    maxs, _ = torch.max(result, dim=2)

    return torch.mean(torch.abs(maxs), dim=1)


def max_amplitude_difference_torch(
    eq: torch.Tensor, denoised_eq: torch.Tensor
) -> float:
    """
    Assumes tensors are of shape (B, C, T) e.g (64, 3, 6120)
    """
    max_eq, _ = torch.max(eq, dim=2)
    max_denoised, _ = torch.max(denoised_eq, dim=2)
    return torch.mean(torch.abs(max_denoised / max_eq), dim=1)


def p_wave_onset_difference_torch(
    eq: torch.Tensor, denoised_eq: torch.Tensor, shift: torch.Tensor
) -> float:
    """
    Assumes eq.shape == (B, C, T) e.g (64, 3, 6120)
    """

    def z_detect_pytorch(a, nsta):
        sta = torch.cumsum(a**2, dim=-1)  # (B, C, T)
        sta[:, :, nsta + 1 :] = sta[:, :, nsta:-1] - sta[:, :, : -nsta - 1]
        sta[:, :, nsta] = sta[:, :, nsta - 1]
        sta[:, :, :nsta] = 0
        a_mean = torch.mean(sta, dim=-1, keepdim=True)  # (B, C, 1)
        a_std = torch.std(sta, dim=-1, keepdim=True)  # (B, C, 1)
        _z = (sta - a_mean) / (a_std + 1e-12)  # (B, C, T)
        return _z

    def find_onset_pytorch(denoised_eq, threshold=0.05, nsta=20):
        zfunc = z_detect_pytorch(denoised_eq, nsta=nsta)  # (B, C, T)
        zfunc -= torch.mean(zfunc[:, :, :500], dim=-1, keepdim=True)
        zfunc = zfunc[:, :, 700:]  # (B, C, T - 700)
        zfunc = torch.where(zfunc > threshold, zfunc, torch.tensor(float("-inf")))
        return torch.argmax(zfunc, dim=-1) + 700

    ground_truth = 6000 - shift  # (B,)
    ground_truth = ground_truth.unsqueeze(1)  # (B, 1)
    denoised_p_wave_onset = find_onset_pytorch(denoised_eq).float()  # (B, C)

    return torch.mean(torch.abs(ground_truth - denoised_p_wave_onset), dim=-1)

import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
import einops

from obspy.signal.cross_correlation import correlate
from obspy.signal.trigger import z_detect

# ======================================= NUMPY METRICS ===================================================


def cross_correlation(eq: ndarray, denoised_eq: ndarray, shift: int = 0) -> float:
    """
    Computes the cross correlation of clean earthquake and denoised earthquake
    Args:
        eq (ndarray): clean earthquake numpy array with shape (B, C, T)
        denoised_eq (ndarray): denoised earthquake numpy array with shape (B, C, T)
    Returns:
        float: mean of the maximum cross correlation of the channels
    """
    B, C, T = eq.shape

    # Store cross-correlations for each channel
    channel_correlations = []

    for c in range(C):
        # Normalize each channel by its max value to avoid amplitude bias
        max_eq_channel = np.max(np.abs(eq[:, c, :]), axis=1, keepdims=True) + 1e-12
        max_denoised_channel = (
            np.max(np.abs(denoised_eq[:, c, :]), axis=1, keepdims=True) + 1e-12
        )

        eq_channel = eq[:, c, :] / max_eq_channel
        denoised_eq_channel = denoised_eq[:, c, :] / max_denoised_channel

        corr = np.array(
            [
                correlate(eq_channel[i], denoised_eq_channel[i], shift=shift)
                for i in range(len(eq))
            ]
        )

        # Take the maximum correlation for this channel
        channel_correlations.append(np.max(corr, axis=1, keepdims=True))

    ccs = np.concatenate(channel_correlations, axis=1)
    mean = np.mean(ccs, axis=1)

    # Return the mean of cross-correlations across channels
    return mean


def max_amplitude_difference(eq: ndarray, denoised_eq: ndarray) -> float:
    """
    Computes the maximum amplitdue difference of clean earthquake and denoised earthquake
    Args:
        eq (ndarray): clean earthquake numpy array with shape (B, C, T)
        denoised_eq (ndarray): denoised earthquake numpy array with shape (B, C, T)
    Returns:
        float: mean of maximum amplitude difference of the channels
    """

    channel_max_ratios = []
    for c in range(eq.shape[1]):
        max_eq_channel = np.max(eq[:, c, :], axis=1, keepdims=True) + 1e-12
        max_denoised_channel = (
            np.max(denoised_eq[:, c, :], axis=1, keepdims=True) + 1e-12
        )
        channel_max_ratios.append(np.abs(max_denoised_channel / max_eq_channel))

    channel_max_ratios = np.concatenate(channel_max_ratios, axis=1)

    return np.mean(channel_max_ratios, axis=1)


def find_onset(denoised_eq, threshold=0.05, nsta=20):
    zfunc = z_detect(denoised_eq, nsta=nsta)
    zfunc -= np.mean(zfunc[:500])
    return np.argmax(zfunc[700:] > threshold) + 700


def p_wave_onset_difference(eq: ndarray, denoised_eq: ndarray, shift: int) -> float:
    """
    Computes the p-wave onset difference of clean earthquake and denoised earthquake
    Args:
        eq (ndarray): clean earthquake numpy array with shape (B, C, T)
        denoised_eq (ndarray): denoised earthquake numpy array with shape (B, C, T)
    Returns:
        float: mean of the p-wave onset difference of the channels
    """
    ground_truth = 6000 - shift

    # Compute P wave onset for each channel
    channel_onset_diffs = []
    for c in range(denoised_eq.shape[1]):
        channel = denoised_eq[:, c, :]
        denoised_p_wave_onset = np.array(
            [find_onset(channel[i]) for i in range(len(channel))]
        )
        channel_onset_diffs.append(
            np.abs(ground_truth - denoised_p_wave_onset)[:, np.newaxis]
        )

    channel_onset_diffs = np.concatenate(channel_onset_diffs, axis=1)

    # Return mean of channel-wise onset differences
    return np.mean(channel_onset_diffs, axis=1)


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

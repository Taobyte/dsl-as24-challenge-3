import numpy as np
from numpy import ndarray
import keras 
import torch 
import torch.nn.functional as F
import einops

from obspy.signal.cross_correlation import correlate
from obspy.signal.trigger import z_detect


def cross_correlation(eq: ndarray, denoised_eq: ndarray,shift=0) -> float:
    max_eq = np.max(eq) + 1e-12
    max_denoised = np.max(denoised_eq) + 1e-12
    corr = correlate(eq / max_eq, denoised_eq / max_denoised, shift=shift)
    return np.max(corr)

"""
def cross_correlation(eq: ndarray, denoised_eq: ndarray, event_shift:int) -> float:
    idx_start = 6000-event_shift-500
    idx_end = 6000-event_shift+1000
    corr = np.correlate(eq[idx_start:idx_end], denoised_eq[idx_start:idx_end])
    return corr
"""

def max_amplitude_difference(eq: ndarray, denoised_eq: ndarray) -> float:
    max_eq = keras.ops.max(eq) + 1e-12
    max_denoised = keras.ops.max(denoised_eq)
    return keras.ops.abs(max_denoised / max_eq)


def find_onset(denoised_eq,threshold=0.05,nsta=20):
    zfunc = z_detect(denoised_eq, nsta=nsta)
    zfunc -= np.mean(zfunc[:500])
    return np.argmax(zfunc[700:]>threshold) + 700 

def p_wave_onset_difference(eq: ndarray, denoised_eq: ndarray, shift: int) -> float:
    
    ground_truth = 6000 - shift
    denoised_p_wave_onset = find_onset(denoised_eq)

    return np.abs(ground_truth - denoised_p_wave_onset)



class CCMetric(keras.metrics.Metric):

    def __init__(self, name='cross_correlation_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cc = self.add_variable(
            shape=(),
            initializer='zeros',
            name='cross_correlation'
        )

    def update_state(self, y_true: ndarray, y_pred: ndarray) -> None:
        
        cc = cross_correlation(y_true, y_pred)
        self.cc.assign(cc)

    def result(self):
        return self.cc

class AmpMetric(keras.metrics.Metric):

    def __init__(self, name='amplitude_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.amp_ratio = self.add_variable(
            shape=(),
            initializer='zeros',
            name='max_amplitude_ratio'
        )

    def update_state(self, y_true: ndarray, y_pred: ndarray, sample_weight=None) -> None:

        print(self.amp_ratio)
        amp_ratio = max_amplitude_difference(y_true, y_pred)
        print(amp_ratio) 
        print(y_true.shape)
        print(y_pred.shape)
        self.amp_ratio.assign(amp_ratio)

    def result(self) -> float:
        return self.amp_ratio

# Not correctly implemented yet
class PWaveMetric(keras.metrics.Metric):

    def __init__(self, name='p_wave_onset_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.p_wave_onset = self.add_variable(
            shape=(),
            initializer='zeros',
            name='p_wave_onset'
        )

    def update_state(self, y_true: ndarray, y_pred: ndarray) -> None:
        # TODO: implement tuple (signal, shift) input in DeepDenoiser
        p_wave_onset_diff = p_wave_onset_difference(y_true, y_pred)
        self.p_wave_onset.assign(p_wave_onset_diff)

    def result(self):
        return self.cc

# ======================================= TORCH METRICS ==================================================

def cross_correlation_torch(eq: torch.Tensor, denoised_eq: torch.Tensor) -> torch.Tensor:
    """
    Pytorch's Conv1d is implemented as cross correlation
    """

    B, C, T = eq.shape
    eq -= eq.mean(dim=-1, keepdim=True)
    denoised_eq -= denoised_eq.mean(dim=-1, keepdim=True)
    eq_reshaped = einops.rearrange(eq, "b c t -> (b c) t")
    denoised_eq_reshaped = einops.rearrange(denoised_eq, "b c t -> (b c) t")
    
    result = F.conv1d(eq_reshaped, denoised_eq_reshaped.unsqueeze(1), padding=T-1, groups=B*C)
    result = einops.rearrange(result, "(b channels) t -> b channels t", channels=C)

    # normalization 
    norm = (torch.sum(eq**2, dim=-1, keepdim=True) * torch.sum(denoised_eq**2, dim=-1, keepdim=True))**0.5
    result = result / (norm + 1e-12)
    maxs, _ = torch.max(result, dim=2)

    return torch.mean(torch.abs(maxs), dim=1)

def max_amplitude_difference_torch(eq: torch.Tensor, denoised_eq: torch.Tensor) -> float:
    """
    Assumes tensors are of shape (B, C, T) e.g (64, 3, 6120)
    """
    max_eq, _ = torch.max(eq, dim=2)
    max_denoised, _ = torch.max(denoised_eq, dim=2)
    return torch.mean(torch.abs(max_denoised / max_eq), dim=1)

def p_wave_onset_difference_torch(eq: torch.Tensor, denoised_eq: torch.Tensor, shift: torch.Tensor) -> float:
    """
    Assumes eq.shape == (B, C, T) e.g (64, 3, 6120)
    """

    def z_detect_pytorch(a, nsta):
        sta = torch.cumsum(a ** 2, dim=-1) # (B, C, T)
        sta[:, :, nsta + 1:] = sta[:, :, nsta:-1] - sta[:, :, :-nsta - 1]
        sta[:, :, nsta] = sta[:, :, nsta - 1]
        sta[:, :, :nsta] = 0
        a_mean = torch.mean(sta, dim=-1, keepdim=True) # (B, C, 1)
        a_std = torch.std(sta, dim=-1, keepdim=True) # (B, C, 1)
        _z = (sta - a_mean) / (a_std + 1e-12) # (B, C, T)
        return _z
    
    def find_onset_pytorch(denoised_eq,threshold=0.05,nsta=20):
        zfunc = z_detect_pytorch(denoised_eq, nsta=nsta) # (B, C, T)
        zfunc -= torch.mean(zfunc[:, :, :500], dim=-1, keepdim=True)
        zfunc = zfunc[:, :, 700:] # (B, C, T - 700)
        zfunc = torch.where(zfunc > threshold, zfunc, torch.tensor(float('-inf')))
        return torch.argmax(zfunc, dim=-1) + 700

    ground_truth = 6000 - shift # (B,)
    ground_truth = ground_truth.unsqueeze(1) # (B, 1)
    denoised_p_wave_onset = find_onset_pytorch(denoised_eq).float() # (B, C)

    return torch.mean(torch.abs(ground_truth - denoised_p_wave_onset), dim=-1)

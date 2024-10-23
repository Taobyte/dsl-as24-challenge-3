import numpy as np
from numpy import ndarray


def cross_correlation(eq: ndarray, denoised_eq: ndarray, event_shift:int) -> float:
    idx_start = 6000-event_shift-500
    idx_end = 6000-event_shift+1000
    corr = np.correlate(eq[idx_start:idx_end], denoised_eq[idx_start:idx_end])
    return corr

def max_amplitude_difference(eq: ndarray, denoised_eq: ndarray) -> float:
    max_eq = np.max(eq)
    max_denoised = np.max(denoised_eq)
    return np.abs(max_denoised / max_eq)

def p_wave_onset_difference(eq: ndarray, denoised_eq: ndarray, shift: int) -> float:
    return 0.0
import numpy as np
from numpy import ndarray
from obspy.signal.cross_correlation import correlate
from obspy.signal.trigger import z_detect


def cross_correlation(eq: ndarray, denoised_eq: ndarray,shift=0) -> float:
    return np.max(correlate(eq / np.max(eq), denoised_eq / np.max(denoised_eq), shift=shift))

"""
def cross_correlation(eq: ndarray, denoised_eq: ndarray, event_shift:int) -> float:
    idx_start = 6000-event_shift-500
    idx_end = 6000-event_shift+1000
    corr = np.correlate(eq[idx_start:idx_end], denoised_eq[idx_start:idx_end])
    return corr
"""

def max_amplitude_difference(eq: ndarray, denoised_eq: ndarray) -> float:
    max_eq = np.max(eq)
    max_denoised = np.max(denoised_eq)
    return np.abs(max_denoised / max_eq)


def find_onset(denoised_eq,threshold=0.05,nsta=20):
    zfunc = z_detect(denoised_eq, nsta=nsta)
    zfunc -= np.mean(zfunc[:500])
    return np.argmax(zfunc[700:]>threshold) + 700 

def p_wave_onset_difference(eq: ndarray, denoised_eq: ndarray, shift: int) -> float:
    
    ground_truth = 6000 - shift
    denoised_p_wave_onset = find_onset(denoised_eq)

    return np.abs(ground_truth - denoised_p_wave_onset)

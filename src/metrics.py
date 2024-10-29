import numpy as np
from numpy import ndarray
import keras 

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
    max_eq = np.max(eq) + 1e-12
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

    def update_state(self, y_true: ndarray, y_pred: ndarray) -> None:

        amp_ratio = max_amplitude_difference(y_true, y_pred)
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

import omegaconf
import torch
import numpy as np

from butterworth_filter import bandpass_obspy
from src.dataset import get_dataloaders_pytorch
from src.metrics import (
    cross_correlation,
    max_amplitude_difference,
    p_wave_onset_difference,
)


def get_metrics_butterworth(cfg: omegaconf.DictConfig):
    """get baseline results for butterworth bandpass filter

    Args:
        - assoc: list of (event, noise, snr, shift) associations
        - snr: the signal to noise ratio at which we score
        - idx: which coordinate to score (i.e. 0=Z, 1=N, 2=E)
    Returns:
        - a dictionary with results (mean and std) for the cross-correlation,
          maximum amplitude difference (percentage) and p wave onset shift (timesteps)
    """

    freq_range = cfg.freq_range
    sampling_rate = cfg.sampling_rate

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)
    ccs = []
    amplitudes = []
    onsets = []
    for snr in cfg.snrs:
        for eq, noise, shifts in test_dl:
            noisy_eq = (snr * eq + noise).numpy()
            eq = eq.numpy()
            shifts = shifts.numpy()

            filtered = np.apply_along_axis(
                lambda x: bandpass_obspy(
                    x,
                    freqmin=freq_range[0],
                    freqmax=freq_range[1],
                    df=sampling_rate,
                    corners=4,
                    zerophase=False,
                ),
                axis=-1,
                arr=noisy_eq,
            )
            print(filtered.shape)

        ccs.append(cross_correlation(eq, filtered))
        amplitudes.append(max_amplitude_difference(eq, filtered))
        onsets.append(p_wave_onset_difference(eq, filtered, shift=shifts))

    amplitudes = np.array(amplitudes)

    # cross, SNR, max amplitude difference
    cross_correlation_mean = np.mean(ccs)
    cross_correlation_std = np.std(ccs)
    max_amplitude_difference_mad = np.mean(np.abs(1 - amplitudes))
    max_amplitude_difference_std = np.std(np.abs(1 - amplitudes))
    p_wave_onset_difference_mean = np.mean(onsets)
    p_wave_onset_difference_std = np.std(onsets)

    return [
        cross_correlation_mean,
        cross_correlation_std,
        max_amplitude_difference_mad,
        max_amplitude_difference_std,
        p_wave_onset_difference_mean,
        p_wave_onset_difference_std,
    ]

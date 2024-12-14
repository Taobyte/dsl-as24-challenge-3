import pathlib

import hydra
import omegaconf
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.Butterworth.butterworth_filter import bandpass_obspy
from src.dataset import get_dataloaders_pytorch
from src.metrics import (
    cross_correlation,
    max_amplitude_difference,
    p_wave_onset_difference,
)


def get_metrics_butterworth(cfg: omegaconf.DictConfig) -> None:
    """
    Saves metrics on test set for butterworth filter

    Args:
        - cfg (omegaconf.DictConfig): hydra config
    """
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    freq_range = cfg.freq_range
    sampling_rate = cfg.sampling_rate

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)
    ccs = []
    amplitudes = []
    onsets = []
    for snr in cfg.snrs:
        for eq, noise, shifts in tqdm(test_dl, total=len(test_dl)):
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
            ccs.append(cross_correlation(eq, filtered))
            amplitudes.append(max_amplitude_difference(eq, filtered))
            onsets.append(p_wave_onset_difference(eq, filtered, shift=shifts))

        ccs = np.concatenate(ccs)
        amplitudes = np.concatenate(amplitudes)
        onsets = np.concatenate(onsets)

        snr_metrics = {
            "cross_correlation": ccs,
            "max_amplitude_difference": amplitudes,
            "p_wave_onset_difference": onsets,
        }
        df = pd.DataFrame(snr_metrics)
        df.to_csv(output_dir / f"snr_{snr}_metrics_Butterworth.csv", index=False)

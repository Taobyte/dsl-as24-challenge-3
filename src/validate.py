import os
import tqdm
import pathlib

os.environ["KERAS_BACKEND"] = "jax"
import keras
import hydra
import omegaconf
import numpy as np
import pandas as pd
import torch as th

from data import get_signal_noise_assoc, InputSignals
from metrics import cross_correlation, p_wave_onset_difference, max_amplitude_difference
from utils import Mode


def get_metrics(model: keras.Model, assoc: list, snr: int, cfg:omegaconf.DictConfig, idx:int=0): 
    
    test_dataset = InputSignals(assoc, Mode.TEST, snr)
    test_dl = th.utils.data.DataLoader(test_dataset, cfg.model.batch_size, shuffle=False)
    ccs = []
    amplitudes = []
    onsets = []
    for (noisy_batch, eq_batch, shifts) in test_dl:
        # get predictions
        predicted_mask = model(noisy_batch)
        stft_real, stft_imag = keras.ops.stft(noisy_batch, 100, 24, 126)
        stft = np.concatenate([stft_real, stft_imag], axis=1)
        masked = stft * predicted_mask
        time_domain_result = keras.ops.istft(
            (masked[:, :3, :, :], masked[:, 3:6, :, :]), 100, 24, 126
        )

        # compute metrics
        eq_batch = eq_batch.numpy()
        time_domain_result = np.array(time_domain_result)
        corr = [cross_correlation(a, b) 
                for a, b in zip(eq_batch[:,idx,:], time_domain_result[:,idx,:])]
        ccs.extend(corr)
        max_amplitude_differences = [max_amplitude_difference(a, b) 
                                     for a, b in zip(eq_batch[:,idx,:], time_domain_result[:,idx,:])]
        amplitudes.extend(max_amplitude_differences)
        onset = [p_wave_onset_difference(a,b,shift) 
                 for a,b,shift in zip(eq_batch[:,idx,:], time_domain_result[:,idx,:], shifts)]
        onsets.extend(onset)

    return ccs, amplitudes, onsets

def compute_metrics(cfg: omegaconf.DictConfig) -> pd.DataFrame:

    assoc = get_signal_noise_assoc(cfg.user.data.signal_path, cfg.user.data.noise_path, Mode.TEST, cfg.size_testset)

    predictions = {
        "cc_mean": [],
        "cc_std": [],
        "max_amp_diff_mean": [],
        "max_amp_diff_std": [],
        "p_wave_mean": [],
        "p_wave_std": [],
    }

    model = keras.saving.load_model(cfg.user.model_path)
    print(f"running predictions for snrs {cfg.snrs}")
    for snr in tqdm.tqdm(cfg.snrs, total=len(cfg.snrs)):


        cross_correlations, max_amplitude_differences, p_wave_onset_differences = get_metrics(model, assoc, snr, cfg, idx=0)

        # cross, SNR, max amplitude difference
        cross_correlation_mean = np.mean(cross_correlations)
        cross_correlation_std = np.std(cross_correlations)

        max_amplitude_difference_mean = np.mean(max_amplitude_differences)
        max_amplitude_difference_std = np.std(max_amplitude_differences)

        p_wave_onset_difference_mean = np.mean(p_wave_onset_differences)
        p_wave_onset_difference_std = np.std(p_wave_onset_differences)

        predictions["cc_mean"].append(cross_correlation_mean)
        predictions["cc_std"].append(cross_correlation_std)
        predictions["max_amp_diff_mean"].append(max_amplitude_difference_mean)
        predictions["max_amp_diff_std"].append(max_amplitude_difference_std)
        predictions["p_wave_mean"].append(p_wave_onset_difference_mean)
        predictions["p_wave_std"].append(p_wave_onset_difference_std)

    df = pd.DataFrame(predictions)
    df["snr"] = cfg.snrs
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df.to_csv(output_dir / 'metrics.csv')

    return df
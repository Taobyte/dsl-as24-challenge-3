import omegaconf 
import hydra
import pathlib
import os
from typing import Union

import torch 
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from metrics import cross_correlation, max_amplitude_difference, p_wave_onset_difference
from models.CleanUNet.dataset import CleanUNetDataset


def get_metrics_clean_unet(
    model: keras.Model, cfg: omegaconf.DictConfig, idx: int = 0
):
    """compute metrics for wavedecompnet model

    Args:
        - model: the keras model to score
        - assoc: list of (event, noise, snr, shift) associations
        - snr: the signal to noise ratio at which to score
        - cfg: the config for batch size
        - idx: which coordinate to score (i.e. 0=Z, 1=N, 2=E)
    Returns:
        - a dictionary with results (mean and std) for the cross-correlation,
          maximum amplitude difference (percentage) and p wave onset shift (timesteps)
    """
    test_dataset = CleanUNetDataset(
        cfg.user.data.signal_path + "/validation/",
        cfg.user.data.noise_path + "/validation/",
        cfg.model.signal_length
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset, cfg.model.batch_size, shuffle=False
    )
    ccs = []
    amplitudes = []
    onsets = []
    for noisy_batch, eq_batch, shifts in test_dl:

        # get predictions
        prediction = model(noisy_batch)

        # compute metrics
        eq_batch = eq_batch.numpy()
        prediction = np.array(prediction)
        corr = [
            cross_correlation(a, b)
            for a, b in zip(eq_batch[:, 0, :, idx], prediction[:, 0, :, idx])
        ]
        ccs.extend(corr)
        max_amplitude_differences = [
            max_amplitude_difference(a, b)
            for a, b in zip(eq_batch[:, 0, :, idx], prediction[:, 0, :, idx])
        ]
        amplitudes.extend(max_amplitude_differences)
        onset = [
            p_wave_onset_difference(a, b, shift)
            for a, b, shift in zip(
                eq_batch[:, 0, :, idx], prediction[:, 0, :, idx], shifts
            )
        ]
        onsets.extend(onset)

    return np.array(ccs), np.array(amplitudes), np.array(onsets)



def visualize_predictions_clean_unet(model: Union[str, keras.Model], signal_path: str, noise_path:str, signal_length: int, n_examples: int, snrs:list[int], channel:int = 0, epoch="") -> None:
    
    print("Visualizing predictions")
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    epoch_dir = os.path.join(output_dir, str(epoch))
    os.makedirs(epoch_dir, exist_ok=True)

    if isinstance(model, str):
        model = keras.saving.load_model(model)

    for snr in tqdm(snrs, total=len(snrs)):

        test_dl = torch.utils.data.DataLoader(CleanUNetDataset(signal_path + "/validation", noise_path + "/validation", signal_length, snr, snr), batch_size=n_examples)
        input, ground_truth = next(iter(test_dl))
        predictions = model(input)

        _, axs = plt.subplots(n_examples, 3,  figsize=(15, n_examples * 3))
        time = range(signal_length)

        for i in tqdm(range(n_examples), total=n_examples):

            axs[i,0].plot(time, input[i,:,channel]) # noisy earthquake
            axs[i,1].plot(time, ground_truth[i,:,channel]) # ground truth noise
            axs[i,2].plot(time, predictions[i,:,channel]) # predicted noise

            """
            row_y_values = []
            row_y_values.extend(input[i, :, channel].numpy())
            row_y_values.extend(ground_truth[i, :, channel].numpy())
            row_y_values.extend(predictions[i, :, channel].numpy())
            
            # Get the y-axis limits for this row
            y_min = np.min(row_y_values)
            y_max = np.max(row_y_values)
            """
            for j in range(3):
                axs[i, j].set_ylim(-2, 2)
        
        column_titles = ["Noisy Earthquake", "Ground Truth Signal", "Prediction"]
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(epoch_dir + f'/visualization_snr_{snr}.png')
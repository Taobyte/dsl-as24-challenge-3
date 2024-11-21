import omegaconf
import hydra
import pathlib 

import keras 
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from metrics import cross_correlation, max_amplitude_difference, p_wave_onset_difference
from utils import Mode
from models.ColdDiffusion.dataset import ColdDiffusionDataset, TestColdDiffusionDataset

def get_metrics_cold_diffusion(
    model: keras.Model, snr: int, cfg: omegaconf.DictConfig, idx: int = 0,
):
    """compute metrics for colddiffusion model

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
    test_dataset = TestColdDiffusionDataset(
        cfg.user.data.test_data_file
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, cfg.model.batch_size, shuffle=False
    )
    ccs = []
    amplitudes = []
    onsets = []
    for eq_batch, noise_batch, shifts in test_dl:

        # get predictions
        t = np.ones((noise_batch.shape[0],)) * 50
        noisy_batch = eq_batch * snr + noise_batch
        prediction = model(noisy_batch, t, training=False)

        # compute metrics
        eq_batch = eq_batch.numpy()
        prediction = np.array(prediction)
        corr = [
            cross_correlation(a, b)
            for a, b in zip(eq_batch[:, :, idx], prediction[:, :, idx])
        ]
        ccs.extend(corr)
        max_amplitude_differences = [
            max_amplitude_difference(a, b)
            for a, b in zip(eq_batch[:, :, idx], prediction[:, :, idx])
        ]
        amplitudes.extend(max_amplitude_differences)
        onset = [
            p_wave_onset_difference(a, b, shift)
            for a, b, shift in zip(
                eq_batch[:, :, idx], prediction[:, :, idx], shifts
            )
        ]
        onsets.extend(onset)

    return np.array(ccs), np.array(amplitudes), np.array(onsets)


def visualize_predictions_cold_diffusion(cfg):
    
    print("Visualizing predictions")
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):

        n_examples = cfg.user.plot_n
        signal_length = cfg.model.signal_length
        channel = cfg.user.plot_channel
        test_dataset = TestColdDiffusionDataset(
            cfg.user.data.test_data_file
        )
        test_dl = torch.utils.data.DataLoader(
            test_dataset, cfg.model.test_batch_size, shuffle=False
        )
        model = keras.saving.load_model(cfg.user.test_model_path)
        eq, noise, _ = next(iter(test_dl))
        print("shapes ", eq.shape, noise.shape)
        noisy = eq*snr + noise
        ground_truth = eq
        t = np.ones((noise.shape[0],)) * 50
        predictions = model(noisy, t, training=False).detach()

        _, axs = plt.subplots(n_examples, 3,  figsize=(15, n_examples * 3))
        time = range(signal_length)

        for i in tqdm(range(n_examples), total=n_examples):

            axs[i,0].plot(time, noisy[i,channel,:]) # noisy earthquake
            axs[i,1].plot(time, ground_truth[i,channel,:]) # ground truth noise
            axs[i,2].plot(time, predictions[i,channel,:]) # predicted noise

            row_y_values = []
            row_y_values.extend(noisy[i, channel, :].numpy())
            row_y_values.extend(ground_truth[i, channel, :].numpy())
            row_y_values.extend(predictions[i, channel, :].numpy())
            
            # Get the y-axis limits for this row
            y_min = np.min(row_y_values)
            y_max = np.max(row_y_values)

            for j in range(3):
                axs[i, j].set_ylim(y_min, y_max)
        
        column_titles = ["Noisy Earthquake", "Ground Truth Signal", "Prediction"]
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_dir / f'visualization_snr_{snr}.png')
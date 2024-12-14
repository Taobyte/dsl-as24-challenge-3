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
from models.ColdDiffusion.dataset import TestColdDiffusionDataset
import models.ColdDiffusion.utils.testing as testing
from models.ColdDiffusion.train_validate import load_model_and_weights

def get_metrics_cold_diffusion(
    model, snr: int, cfg: omegaconf.DictConfig, idx: int = 0,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testing.initialize_parameters(cfg.model.T)
    model = load_model_and_weights(cfg.user.path_model, cfg)
    model = model.to(device)

    ccs_sample = []
    amplitudes_sample = []
    onsets_sample = []
    ccs = []
    amplitudes = []
    onsets= []

    def compute_metrs(eq_batch, prediction, shifts):
        corr = [
            cross_correlation(a, b)
            for a, b in zip(eq_batch[:, idx, :], prediction[:, idx, :])
        ]
        max_amplitude_differences = [
            max_amplitude_difference(a, b).cpu().numpy()
            for a, b in zip(eq_batch[:, idx, :], prediction[:, idx, :])
        ]
        onset = [
            p_wave_onset_difference(a, b, shift)
            for a, b, shift in zip(
                eq_batch[:, idx, :], prediction[:, idx, :], shifts
            )
        ]
        return corr, max_amplitude_differences, onset

    with torch.no_grad():
        for eq_batch, noise_batch, shifts in test_dl:

            # get predictions
            eq_batch = eq_batch * snr
            eq_in = eq_batch.to(device)
            noise_real = noise_batch.to(device)
            signal_noisy = eq_in + noise_real
            
            # compute metrics
            shifts = np.array(shifts, dtype=int)

            if not cfg.model.sampling:
                t = torch.Tensor([cfg.model.T - 1]).long().to(device)
                restored_dir = testing.direct_denoising(model, signal_noisy.to(device).float(), t).cpu()
                corr, max_amplitude_differences, onset = compute_metrs(eq_batch.numpy(), restored_dir.numpy(), shifts)
                ccs.append(corr)
                amplitudes.extend(max_amplitude_differences)
                onsets.extend(onset)
                return np.array(ccs), np.array(amplitudes), np.array(onsets)
            else:
                t = cfg.model.T - 1
                restored_sample = testing.sample(
                                                model,
                                                signal_noisy.float(),
                                                t,
                                                batch_size=signal_noisy.shape[0]
                                                ).cpu()
                corr, max_amplitude_differences, onset = compute_metrs(eq_batch.numpy(), restored_sample.numpy(), shifts)
                ccs_sample.extend(corr)
                amplitudes_sample.extend(max_amplitude_differences)
                onsets_sample.extend(onset)
                return np.array(ccs_sample), np.array(amplitudes_sample), np.array(onsets_sample)


def visualize_predictions_cold_diffusion(cfg):
    
    print("Visualizing predictions")
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    test_dataset = TestColdDiffusionDataset(
        cfg.user.data.test_data_file
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, cfg.model.test_batch_size, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testing.initialize_parameters(cfg.model.T)
    model = load_model_and_weights(cfg.user.path_model, cfg)
    model = model.to(device)

    with torch.no_grad():
        for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):

            n_examples = cfg.user.plot_n
            signal_length = cfg.model.signal_length
            channel = cfg.user.plot_channel
            eq, noise, _ = next(iter(test_dl))
            noisy = eq * snr + noise
            ground_truth = eq * snr
            t = torch.Tensor([cfg.model.T - 1]).long().to(device)
            
            restored_dir = testing.direct_denoising(model, noisy.to(device).float(), t).cpu()

            t = cfg.model.T - 1
            restored_sample = testing.sample(
                                            model,
                                            noisy.to(device).float(),
                                            t,
                                            batch_size=noisy.shape[0]
                                            ).cpu()
            
            # take one
            if cfg.model.sampling:
                predictions = restored_sample
            else:
                predictions = restored_dir

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

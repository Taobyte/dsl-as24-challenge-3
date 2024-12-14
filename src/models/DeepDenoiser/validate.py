import pathlib
import logging

import omegaconf
from omegaconf import OmegaConf
import hydra

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import get_trained_model, Model
from src.metrics import (
    cross_correlation_torch,
    max_amplitude_difference_torch,
    p_wave_onset_difference_torch,
)
from src.dataset import get_dataloaders_pytorch
from src.stft import get_stft, get_istft, get_mask

logger = logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_metrics_deepdenoiser(cfg: omegaconf.DictConfig):
    logger.info("Computing metrics on testset for CleanUNet.")
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    model, config = get_trained_model(cfg, Model.DeepDenoiser)
    n_fft = config.model.architecture.n_fft
    hop_length = config.model.architecture.hop_length
    win_length = config.model.architecture.win_length
    trace_length = config.trace_length

    for snr in cfg.snrs:
        test_dl = get_dataloaders_pytorch(cfg, return_test=True)
        ccs, amplitudes, onsets = [], [], []
        with torch.no_grad():
            for eq, noise, shifts in tqdm(test_dl, total=len(test_dl)):
                eq = eq.float().to(device)
                noise = noise.float().to(device)
                shifts = torch.from_numpy(np.array(shifts)).to(device)
                noisy_eq = snr * eq + noise
                stft_noisy_eq = get_stft(noisy_eq, n_fft, hop_length, win_length)

                mask = torch.nn.functional.sigmoid(model(noisy_eq))
                denoised_mask = stft_noisy_eq * mask
                prediction = get_istft(
                    denoised_mask, n_fft, hop_length, win_length, trace_length
                )

                ccs.append(cross_correlation_torch(eq, prediction))
                amplitudes.append(max_amplitude_difference_torch(eq, prediction))
                onsets.append(p_wave_onset_difference_torch(eq, prediction, shifts))

            ccs = torch.concatenate(ccs, dim=0)
            amplitudes = torch.concatenate(amplitudes, dim=0)
            onsets = torch.concatenate(onsets, dim=0)
            snr_metrics = {
                "cross_correlation": ccs.cpu().numpy(),
                "max_amplitude_difference": amplitudes.cpu().numpy(),
                "p_wave_onset_difference": onsets.cpu().numpy(),
            }
            df = pd.DataFrame(snr_metrics)
            df.to_csv(output_dir / f"snr_{snr}_metrics_DeepDenoiser.csv", index=False)

    return df


def get_predictions_deepdenoiser(
    eq: torch.Tensor, noise: torch.Tensor, cfg: omegaconf.DictConfig
):
    eq = eq.float()
    noise = noise.float()

    trace_length = cfg.trace_length
    n_fft = cfg.model.architecture.n_fft
    hop_length = cfg.model.architecture.hop_length
    win_length = cfg.model.architecture.win_length

    model = get_trained_model(cfg, Model.DeepDenoiser)

    noisy_eq = eq + noise
    stft_noisy_eq = get_stft(noisy_eq, n_fft, hop_length, win_length)

    model.eval()

    with torch.no_grad():
        mask = torch.nn.functional.sigmoid(model(noisy_eq))
    masked_stft = stft_noisy_eq * mask

    istft = get_istft(masked_stft, n_fft, hop_length, win_length, trace_length)

    return istft


def plot_spectograms(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    n_fft = cfg.model.architecture.n_fft
    hop_length = cfg.model.architecture.hop_length
    win_length = cfg.model.architecture.win_length

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)
    model = get_trained_model(cfg, Model.DeepDenoiser)
    model.eval()
    with torch.no_grad():
        for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
            eq, noise, shift = next(iter(test_dl))
            eq, noise = eq.float(), noise.float()
            noisy_eq = eq + noise
            predictions = model(noisy_eq)

            mask = get_mask(eq, noise, n_fft, hop_length, win_length)

            fig, axs = plt.subplots(cfg.plot.n_examples, 3, figsize=(12, 8))

            for i in range(cfg.plot.n_examples):
                axs[i, 0].plot(range(cfg.trace_length), noisy_eq[i, 0, :])
                im1 = axs[i, 1].imshow(mask[i, 0, :, :], cmap="viridis", aspect="auto")
                _ = axs[i, 2].imshow(
                    predictions[i, 0, :, :], cmap="viridis", aspect="auto"
                )

            column_titles = ["Noisy Earthquake", "Ground Truth Mask", "Predicted Mask"]
            for j, title in enumerate(column_titles):
                axs[0, j].set_title(title)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im1, cax=cbar_ax)
            cbar.set_label("Value")

            plt.suptitle(f"SNR: {snr}", fontsize=16)
            plt.savefig(output_dir / f"visualization_snr_{snr}.png")
            plt.close(fig)


def visualize_predictions_deepdenoiser(cfg: omegaconf.DictConfig) -> None:
    logger.info("Visualizing predictions for DeepDenoiser")
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    n_examples = cfg.plot.n_examples
    channel_idx = cfg.plot.channel_idx
    trace_length = cfg.trace_length
    n_fft = cfg.n_fft
    hop_length = cfg.hop_length
    win_length = cfg.win_length

    model = get_trained_model(cfg, Model.DeepDenoiser)

    for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
        test_dl = get_dataloaders_pytorch(cfg, return_test=True)
        eq, noise, _ = next(iter(test_dl))
        eq, noise = eq.float(), noise.float()
        noisy_eq = snr * eq + noise
        stft_noisy_eq = get_stft(noisy_eq, n_fft, hop_length, win_length)
        with torch.no_grad():
            mask = torch.nn.functional.sigmoid(model(noisy_eq))

        predictions = get_istft(
            stft_noisy_eq * mask, n_fft, hop_length, win_length, trace_length
        )

        _, axs = plt.subplots(n_examples, 3, figsize=(15, n_examples * 3))
        time = range(cfg.trace_length)
        for i in tqdm(range(n_examples), total=n_examples):
            axs[i, 0].plot(time, noisy_eq[i, channel_idx, :])
            axs[i, 1].plot(time, eq[i, channel_idx, :])
            axs[i, 2].plot(time, predictions.numpy()[i, channel_idx, :])

            for j in range(3):
                axs[i, j].set_ylim(-2, 2)

        column_titles = ["Noisy Earthquake", "Ground Truth Signal", "Prediction"]
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_dir / f"visualization_snr_{snr}.png")

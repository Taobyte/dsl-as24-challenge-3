import logging

import omegaconf
import hydra
import pathlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import get_trained_model, Model
from src.metrics import (
    cross_correlation_torch,
    max_amplitude_difference_torch,
    p_wave_onset_difference_torch,
)
from src.dataset import get_dataloaders_pytorch

logger = logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_metrics_clean_unet(cfg: omegaconf.DictConfig) -> pd.DataFrame:
    logger.info("Computing metrics on testset for CleanUNet.")
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    model, config = get_trained_model(cfg, Model.CleanUNet)

    for snr in cfg.snrs:
        test_dl = get_dataloaders_pytorch(cfg, return_test=True)
        ccs, amplitudes, onsets = [], [], []
        with torch.no_grad():
            for eq, noise, shifts in tqdm(test_dl, total=len(test_dl)):
                eq = eq.float().to(device)
                noise = noise.float().to(device)
                shifts = torch.from_numpy(np.array(shifts)).to(device)
                noisy_eq = snr * eq + noise
                eq = snr * eq

                prediction = model(noisy_eq)
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
            df.to_csv(output_dir / f"snr_{snr}_metrics_CleanUNet.csv", index=False)

    return df


def get_predictions_cleanunet(
    noisy_eq: torch.Tensor, cfg: omegaconf.DictConfig
) -> torch.Tensor:
    logger.info("Make predictions with CleanUNet.")

    model, config = get_trained_model(cfg, Model.CleanUNetTransformer)

    with torch.no_grad():
        predictions = model(noisy_eq.to(device))

    return predictions.cpu()


def visualize_predictions_clean_unet(cfg: omegaconf.DictConfig) -> None:
    logger.info("Visualizing predictions for CleanUNet")
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    n_examples = cfg.plot.n_examples
    channel_idx = cfg.plot.channel_idx

    model, config = get_trained_model(cfg, Model.CleanUNet)

    for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
        test_dl = get_dataloaders_pytorch(cfg, return_test=True)
        eq, noise, _ = next(iter(test_dl))
        eq, noise = eq.float(), noise.float()
        noisy_eq = snr * eq + noise
        with torch.no_grad():
            predictions = model(noisy_eq)

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

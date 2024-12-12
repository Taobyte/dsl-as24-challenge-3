import omegaconf
import hydra
import pathlib

import torch
from torch import Tensor
import einops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.DeepDenoiser.dataset import get_dataloaders_pytorch
from src.models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser

def get_metrics_deepdenoiser():
    pass


def get_mask(eq: Tensor, noise:Tensor, cfg: omegaconf.DictConfig) -> Tensor:
    B, C, T = eq.shape
    window = torch.hann_window(cfg.model.architecture.win_length)
    eq = einops.rearrange(eq, 'b c t -> (b c) t', b=B, c=C)
    eq_stft = torch.stft(
            eq,
            cfg.model.architecture.n_fft,
            cfg.model.architecture.hop_length,
            cfg.model.architecture.win_length,
            window,
            return_complex=True,
        )
    stft_eq = torch.view_as_real(eq_stft)
    stft_eq = einops.rearrange(stft_eq, "(b c) w h f -> b (c f) w h", b=B, c=C, f=2)

    noise = einops.rearrange(noise, 'b c t -> (b c) t', b=B, c=C)
    noise_stft = torch.stft(
        noise,
        cfg.model.architecture.n_fft,
        cfg.model.architecture.hop_length,
        cfg.model.architecture.win_length,
        window,
        return_complex=True,
    )
    stft_noise = torch.view_as_real(noise_stft)
    stft_noise = einops.rearrange(stft_noise, "(b c) w h f -> b (c f) w h",  b=B, c=C, f=2)

    mask = stft_eq.abs() / (stft_noise.abs() + stft_eq.abs() + 1e-12)

    return mask

def plot_spectograms(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)
    model = DeepDenoiser(**cfg.model.architecture)
    checkpoint = torch.load(
        cfg.user.model_path, map_location=torch.device("cpu"), weights_only=False
    )
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
            eq, noise = next(iter(test_dl))
            eq = eq.float()
            noise = noise.float()
            noisy_eq = eq + noise
            predictions = model(noisy_eq)

            mask = get_mask(eq, noise, cfg)

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
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            plt.savefig(output_dir / f"visualization_snr_{snr}.png")
            plt.close(fig)

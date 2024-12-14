import omegaconf
from omegaconf import OmegaConf
import hydra
import pathlib

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.DeepDenoiser.dataset import get_dataloaders_pytorch
from src.models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser
from src.stft import get_stft, get_istft, get_mask


def get_metrics_deepdenoiser():
    pass


def get_predictions_deepdenoiser(
    eq: torch.Tensor, noise: torch.Tensor, cfg: omegaconf.DictConfig
):
    eq = eq.float()
    noise = noise.float()
    config_path = cfg.user.deep_denoiser_folder + "/.hydra/config.yaml"
    config = OmegaConf.load(config_path)

    trace_length = config.trace_length
    n_fft = config.model.architecture.n_fft
    hop_length = config.model.architecture.hop_length
    win_length = config.model.architecture.win_length

    noisy_eq = eq + noise
    stft_eq = get_stft(eq, n_fft, hop_length, win_length)

    model = DeepDenoiser(**config.model.architecture)
    model.eval()

    with torch.no_grad():
        mask = model(noisy_eq)
    masked_stft = stft_eq * mask

    istft = get_istft(masked_stft, n_fft, hop_length, win_length, trace_length)

    # print(istft.min())
    # print(istft.max())

    return istft


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

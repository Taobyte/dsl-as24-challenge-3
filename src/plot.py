import pathlib
import hydra
import omegaconf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import Model
from src.models.CleanUNet.validate import visualize_predictions_clean_unet
from src.models.DeepDenoiser.validate import (
    plot_spectograms,
    visualize_predictions_deepdenoiser,
)


def visualize_predictions(cfg: omegaconf.DictConfig):
    if cfg.model.model_name == Model.DeepDenoiser.value:
        # plot_spectograms(cfg)
        visualize_predictions_deepdenoiser(cfg)
    elif cfg.model.model_name == Model.CleanUNet.value:
        visualize_predictions_clean_unet(cfg)
    elif cfg.model.model_name == Model.CleanUNet2.value:
        raise NotImplementedError
    elif cfg.model.model_name == Model.CleanSpecNet.value:
        raise NotImplementedError
    else:
        raise NotImplementedError


def metrics_plot(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    models = cfg.plot.models
    metrics_folder_path = cfg.user.metrics_folder
    snrs = cfg.snrs

    dataframes = {}
    column_names = [
        "snr",
        "cc_mean",
        "max_amp_diff_mean",
        "p_wave_mean",
        "cc_std",
        "max_amp_diff_std",
        "p_wave_std",
    ]

    # Compute mean and std for each model and metric
    for model in models:
        rows = []
        for snr in snrs:
            df = pd.read_csv(
                metrics_folder_path + f"/{model}/snr_{snr}_metrics_{model}.csv"
            )
            if len(df.columns) == 4:
                df = df[
                    [
                        "cross_correlation",
                        "max_amplitude_difference",
                        "p_wave_onset_difference",
                    ]
                ]
            mean = df.mean()
            std = df.std()
            row = [snr] + list(mean.values) + list(std.values)
            rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        dataframes[model] = df

    # Define the metrics and their labels
    metrics = [
        ("cc_mean", "cc_std", "Cross Correlation (CC)"),
        ("max_amp_diff_mean", "max_amp_diff_std", "Max Amplitude Diff (MAD)"),
        ("p_wave_mean", "p_wave_std", "P Wave Onset"),
    ]

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("muted", len(models))
    ecolors = sns.color_palette("pastel", len(models))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Plot each metric
    for i, (ax, (mean_col, std_col, ylabel)) in enumerate(zip(axs, metrics)):
        for j, model in enumerate(models):
            df = dataframes[model]
            ax.errorbar(
                df["snr"],
                df[mean_col],
                yerr=df[std_col],
                fmt="o-",
                capsize=5,
                label=models[j],
                color=colors[j],
                ecolor=ecolors[j],
                elinewidth=2,
                capthick=2,
                markersize=8,
            )

        ax.set_xlabel("Signal to Noise Ratio (SNR)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)

        if i == 1:
            ax.set_yscale("log")
        elif i == 0:
            ax.set_ylim(0, 1)

    fig.suptitle("Comparison of Denoising Models", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "metrics.png")
    plt.show()


def overlay_plot(cfg: omegaconf.DictConfig):
    """
    Creates an overlay plot of noisy earthquake, original, and filtered signals

    Args:
        cfg (omegaconf.DictConfig): Configuration dictionary
    """
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("muted", 5)

    channel_idx = cfg.plot.channel_idx
    trace_length = cfg.trace_length

    for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
        data = np.load(cfg.user.prediction_path + f"/snr_{snr}_predictions.npz")
        n = len(data["noisy_eq"])
        fig, axs = plt.subplots(n, 5, figsize=(18, 6 * 5), dpi=300)
        for i in range(n):
            print(data["shift"][i])
            lower_bound = max(0, 6000 - data["shift"][i] - 200)
            upper_bound = min(4096, 6000 - data["shift"][i] + 500)
            length = upper_bound - lower_bound  # length = range(trace_length)
            length = range(length)

            axs[i, 0].plot(
                length,
                data["noisy_eq"][i, channel_idx, lower_bound:upper_bound],
                linewidth=2,
                color=colors[4],
            )
            axs[i, 0].set_title(
                "Noisy Earthquake Signal", fontsize=12, fontweight="bold"
            )
            axs[i, 0].set_ylim(-2, 2)
            axs[i, 0].set_xlabel("Time", fontsize=10)
            axs[i, 0].set_ylabel("Amplitude", fontsize=10)

            axs[i, 1].plot(
                length,
                data["eq"][i, channel_idx, lower_bound:upper_bound],
                linewidth=2,
                color=colors[4],
            )
            axs[i, 1].set_title(
                "Original Earthquake Signal", fontsize=12, fontweight="bold"
            )
            axs[i, 1].set_ylim(-2, 2)
            axs[i, 1].set_xlabel("Time", fontsize=10)
            axs[i, 1].set_ylabel("Amplitude", fontsize=10)

            axs[i, 2].plot(
                length,
                data["butterworth"][i, channel_idx, lower_bound:upper_bound],
                color=colors[0],
                linewidth=2,
                label="Butterworth",
            )
            axs[i, 2].plot(
                length,
                data["deepdenoiser"][i, channel_idx, lower_bound:upper_bound],
                color=colors[1],
                linewidth=2,
                label="DeepDenoiser",
            )
            axs[i, 2].set_title(
                "Filtered Signals Comparison", fontsize=12, fontweight="bold"
            )
            axs[i, 2].set_ylim(-2, 2)
            axs[i, 2].set_xlabel("Time", fontsize=10)
            axs[i, 2].set_ylabel("Normalized Amplitude", fontsize=10)
            axs[i, 2].legend(fontsize=9)

            axs[i, 3].plot(
                length,
                data["butterworth"][i, channel_idx, lower_bound:upper_bound],
                color=colors[0],
                linewidth=2,
                label="Butterworth",
            )
            axs[i, 3].plot(
                length,
                data["cleanunet"][i, channel_idx, lower_bound:upper_bound],
                color=colors[2],
                linewidth=2,
                label="CleanUNet",
            )
            axs[i, 3].set_title(
                "Filtered Signals Comparison", fontsize=12, fontweight="bold"
            )
            axs[i, 3].set_ylim(-2, 2)
            axs[i, 3].set_xlabel("Time", fontsize=10)
            axs[i, 3].set_ylabel("Normalized Amplitude", fontsize=10)
            axs[i, 3].legend(fontsize=9)

            axs[i, 4].plot(
                length,
                data["butterworth"][i, channel_idx, lower_bound:upper_bound],
                color=colors[0],
                linewidth=2,
                label="Butterworth",
            )
            axs[i, 4].plot(
                length,
                data["colddiffusion"][i, channel_idx, lower_bound:upper_bound],
                color=colors[3],
                linewidth=2,
                label="ColdDiffusion",
            )
            axs[i, 4].set_title(
                "Filtered Signals Comparison", fontsize=12, fontweight="bold"
            )
            axs[i, 4].set_ylim(-2, 2)
            axs[i, 4].set_xlabel("Time", fontsize=10)
            axs[i, 4].set_ylabel("Normalized Amplitude", fontsize=10)
            axs[i, 4].legend(fontsize=9)

        plt.tight_layout()
        fig.suptitle(
            f"Earthquake Signal Analysis SNR={snr}",
            fontsize=14,
            fontweight="bold",
            y=1.05,
        )

        output_path = output_dir / f"overlay_plot_snr_{snr}.png"
        fig.savefig(output_path, bbox_inches="tight")

        plt.close(fig)

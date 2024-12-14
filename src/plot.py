import pathlib
import hydra
import omegaconf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import Model
from src.models.CleanUNet.validate import visualize_predictions_clean_unet
from src.models.DeepDenoiser.validate import plot_spectograms


def visualize_predictions(cfg: omegaconf.DictConfig):
    if cfg.model.model_name == Model.DeepDenoiser.value:
        plot_spectograms(cfg)
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
    models = ["Butterworth", "CleanUNet"]
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

    # compute mean and std for each model and metric
    for model in models:
        rows = []
        for snr in snrs:
            df = pd.read_csv(
                metrics_folder_path + f"/CleanUNet/snr_{snr}_metrics_CleanUNet.csv"
            )
            mean = df.mean()
            std = df.std()
            row = [snr] + list(mean.values) + list(std.values)
            rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        dataframes[model] = df

    # Create a figure and axes for 3 subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Define the metrics and their labels
    metrics = [
        ("cc_mean", "cc_std", "Cross Correlation (CC)"),
        ("max_amp_diff_mean", "max_amp_diff_std", "Max Amplitude Diff (MAD)"),
        ("p_wave_mean", "p_wave_std", "P Wave Onset"),
    ]

    colors = ["cornflowerblue", "olivedrab", "tomato"]
    ecolors = ["lightsteelblue", "yellowgreen", "salmon"]

    # Plot each metric
    for i, (ax, (mean_col, std_col, ylabel)) in enumerate(zip(axs, metrics)):
        # Model

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
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=10)

        if i == 1:
            ax.set_yscale("log")
        #     ax.set_ylim(0, 5)
        # elif i == 2:
        #     ax.set_ylim(0, 1000)
        # elif i == 0:
        #     ax.set_ylim(0, 1)

    # Set the title for the entire figure
    fig.suptitle("Comparison of Denoising Models", fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'metrics.png')
    plt.show()

    plt.savefig(output_dir / "metrics.jpg")


def overlay_plot(cfg: omegaconf.DictConfig):
    """
    Creates an overlay plot of noisy earthquake, original, and filtered signals

    Args:
        cfg (omegaconf.DictConfig): Configuration dictionary
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("deep")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    data = np.load(cfg.user.prediction_path)

    channel_idx = cfg.plot.channel_idx
    trace_length = cfg.trace_length

    deepdenoiser_normalized = data["deepdenoiser"][0, channel_idx, :] / np.max(
        np.abs(data["deepdenoiser"][0, channel_idx, :])
    )

    axs[0].plot(
        range(trace_length),
        data["noisy_eq"][0, channel_idx, :],
        linewidth=2,
        color=sns.color_palette("deep")[0],
    )
    axs[0].set_title("Noisy Earthquake Signal", fontsize=12, fontweight="bold")
    axs[0].set_ylim(-2, 2)
    axs[0].set_xlabel("Time", fontsize=10)
    axs[0].set_ylabel("Amplitude", fontsize=10)

    axs[1].plot(
        range(trace_length),
        data["eq"][0, channel_idx, :],
        linewidth=2,
        color=sns.color_palette("deep")[1],
    )
    axs[1].set_title("Original Earthquake Signal", fontsize=12, fontweight="bold")
    axs[1].set_ylim(-2, 2)
    axs[1].set_xlabel("Time", fontsize=10)
    axs[1].set_ylabel("Amplitude", fontsize=10)

    axs[2].plot(
        range(trace_length),
        data["butterworth"][0, channel_idx, :],
        color=sns.color_palette("deep")[2],
        linewidth=2,
        label="Butterworth Filtered",
    )
    axs[2].plot(
        range(trace_length),
        deepdenoiser_normalized,
        color=sns.color_palette("deep")[3],
        linewidth=2,
        label="Deep Denoiser",
    )
    axs[2].set_title("Filtered Signals Comparison", fontsize=12, fontweight="bold")
    axs[2].set_ylim(-2, 2)
    axs[2].set_xlabel("Time", fontsize=10)
    axs[2].set_ylabel("Normalized Amplitude", fontsize=10)
    axs[2].legend(fontsize=9)

    plt.tight_layout()
    fig.suptitle("Earthquake Signal Analysis", fontsize=14, fontweight="bold", y=1.05)

    output_path = output_dir / "overlay_plot.png"
    fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)

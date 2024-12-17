import pathlib
import hydra
import omegaconf
import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator

from src.utils import Model
from src.models.CleanUNet.validate import visualize_predictions_clean_unet
from src.models.DeepDenoiser.validate import (
    plot_spectograms,
    visualize_predictions_deepdenoiser,
)


logger = logging.getLogger()


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
        ("max_amp_diff_mean", "max_amp_diff_std", "Max Amplitude Ratio"),
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

    # fig.suptitle("Comparison of Denoising Models", fontsize=16)
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
    colors = sns.color_palette("muted", 10)

    channel_idx = cfg.plot.channel_idx
    use_overlay = cfg.plot.overlay_plot.use_overlay
    trace_range = None
    if cfg.plot.overlay_plot.range:
        trace_range = list(cfg.plot.overlay_plot.range)
    compare_clean = False
    if cfg.plot.overlay_plot.compare_clean:
        compare_clean = True
    opacity = cfg.plot.overlay_plot.opacity

    if cfg.plot.overlay_plot.specific:
        specific = cfg.plot.overlay_plot.specific
        logger.info(f"Plotting chosen rows: {specific}.")
        n = len(specific)
        fig, axs = plt.subplots(n, 4, figsize=(22, n * 4), dpi=300)
        for i, (idx, snr) in enumerate(specific):
            data = np.load(cfg.user.prediction_path + f"/snr_{snr}_predictions.npz")
            plot_single_row(
                i,
                idx,
                axs,
                data,
                channel_idx,
                colors,
                cfg.trace_length,
                trace_range,
                compare_clean,
                opacity,
                use_overlay,
            )

            axs[i, 0].annotate(
                f"SNR: {snr} ",
                xy=(0, 0.5),
                xytext=(-axs[i, 0].yaxis.labelpad - 15, 0),
                xycoords=axs[i, 0].yaxis.label,
                textcoords="offset points",
                size=12,
                ha="right",
                va="center",
                rotation=90,
                fontweight="bold",
            )
        plt.tight_layout()

        output_path = output_dir / "overlay_plot_specific.png"
        fig.savefig(output_path, bbox_inches="tight")

        plt.close(fig)

    else:
        logger.info(f"Plotting the first {cfg.plot.n_examples} predictions.")
        for snr in tqdm(cfg.snrs, total=len(cfg.snrs)):
            data = np.load(cfg.user.prediction_path + f"/snr_{snr}_predictions.npz")
            n = len(data["noisy_eq"])
            fig, axs = plt.subplots(n, 5, figsize=(18, n * 3), dpi=300)
            for i in range(n):
                plot_single_row(
                    i,
                    i,
                    axs,
                    data,
                    channel_idx,
                    colors,
                    cfg.trace_length,
                    trace_range,
                    compare_clean,
                    opacity,
                    use_overlay,
                )

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


def plot_single_row(
    i: int,
    idx: int,
    axs,
    data: np.ndarray,
    channel_idx: int,
    colors: list,
    trace_length: int,
    trace_range: list[int] = None,
    compare_clean: bool = False,
    opacity: float = 1.0,
    use_overlay: bool = True,
) -> None:
    if trace_range is not None:
        lower_bound = max(0, 6000 - data["shift"][idx] - trace_range[0])
        upper_bound = min(4096, 6000 - data["shift"][idx] + trace_range[1])
    else:
        lower_bound = 0
        upper_bound = trace_length
    delta = upper_bound - lower_bound
    length = range(delta)

    butterworth = data["butterworth"][idx, channel_idx, lower_bound:upper_bound]
    clean_eq = data["eq"][idx, channel_idx, lower_bound:upper_bound]
    label_comparision = "Clean Earthquake" if compare_clean else "Butterworth"

    axs[i, 0].plot(
        length,
        data["noisy_eq"][idx, channel_idx, lower_bound:upper_bound],
        linewidth=2,
        color=colors[0],
        label="Noisy Earthquake",
    )
    axs[i, 0].set_title("Noisy Earthquake", fontsize=12, fontweight="bold")
    """
    axs[i, 1].plot(
        length, clean_eq, linewidth=2, color=colors[0], label="Clean Earthquake"
    )
    axs[i, 1].set_title("Clean Earthquake", fontsize=12, fontweight="bold")
    """

    axs[i, 1].plot(
        length,
        data["deepdenoiser"][idx, channel_idx, lower_bound:upper_bound],
        color=colors[1],
        linewidth=2,
        label="DeepDenoiser",
        alpha=opacity,
    )
    axs[i, 1].set_title("DeepDenoiser", fontsize=12, fontweight="bold")

    axs[i, 3].plot(
        length,
        data["cleanunet"][idx, channel_idx, lower_bound:upper_bound],
        color=colors[2],
        linewidth=2,
        label="CleanUNet",
        alpha=opacity,
    )
    axs[i, 3].set_title("CleanUNet", fontsize=12, fontweight="bold")

    axs[i, 2].plot(
        length,
        data["colddiffusion"][idx, channel_idx, lower_bound:upper_bound],
        color=colors[3],
        linewidth=2,
        label="ColdDiffusion",
        alpha=opacity,
    )
    axs[i, 2].set_title("ColdDiffusion", fontsize=12, fontweight="bold")

    if use_overlay:
        for k in range(3):
            axs[i, (k + 1)].plot(
                length,
                clean_eq if compare_clean else butterworth,
                color=colors[0],
                linewidth=2,
                label=label_comparision,
                alpha=opacity,
            )

    # plot p-wave onset
    for k in range(4):
        axs[i, k].vlines(
            x=trace_range[0]
            if trace_range
            else min(trace_length - 1, 6000 - data["shift"][idx]),
            color=colors[5],
            linestyle="-",
            linewidth=2,
            label="P-Wave Onset",
            ymin=-3.0,
            ymax=3.0,
        )
        if trace_range:
            axs[i, k].set_ylim(-1, 1)
        else:
            axs[i, k].set_ylim(-2, 2)
        axs[i, k].set_xlabel("Time", fontsize=10)
        axs[i, k].set_ylabel("Amplitude", fontsize=10)
        axs[i, k].legend(fontsize="small")


def plot_training_run(cfg: omegaconf.DictConfig):
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    logger.info(
        f"Plotting training & validation loss curves for {cfg.model.model_name}."
    )

    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10")
    train_color, val_color = palette[0], palette[1]

    model_name = cfg.model.model_name
    if model_name == Model.DeepDenoiser.value:
        log_dir = cfg.user.deep_denoiser_folder
    elif model_name == Model.CleanUNet.value:
        log_dir = cfg.user.clean_unet_folder
    else:
        log_dir = ""

    config_path = log_dir + "/.hydra/config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    event_file = [f for f in os.listdir(log_dir) if f.startswith("events")][0]
    event_path = os.path.join(log_dir, event_file)

    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    print("Available Tags:")
    for tag in ea.Tags()["scalars"]:
        print(tag)
    if cfg.model.model_name == Model.DeepDenoiser.value:
        train_loss = ea.Scalars("train_loss")
        val_loss = ea.Scalars("validation_loss")
    else:
        train_loss = ea.Scalars("Train/Train-Loss")
        val_loss = ea.Scalars("Validation/Val-Loss")

    train_steps = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]
    val_steps = [x.step for x in val_loss]
    val_values = [x.value for x in val_loss]

    last_val_epoch = val_steps[-1]
    idx = max(i for i, step in enumerate(train_steps) if step <= last_val_epoch)
    train_steps = train_steps[:idx]
    train_values = train_values[:idx]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(
        train_steps, train_values, label="Training Loss", linewidth=3, color=train_color
    )
    plt.plot(
        val_steps, val_values, label="Validation Loss", linewidth=3, color=val_color
    )

    ax = plt.gca()  # Get current axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if model_name == Model.DeepDenoiser.value:
        ax.set_xticks(np.arange(0, 21, 2))
    elif model_name == Model.CleanUNet.value:
        plt.annotate(
            f"{train_values[-1]:.2f}",
            xy=(train_steps[-1], train_values[-1]),
            xytext=(
                train_steps[-1] + 0.01,
                train_values[-1] + 0.05,
            ),
            # arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=10,
            # fontweight="bold",
            color=train_color,
        )

        plt.annotate(
            f"{val_values[-1]:.2f}",
            xy=(val_steps[-1], val_values[-1]),
            xytext=(
                val_steps[-1] + 0.01,
                val_values[-1] + 0.05,
            ),
            fontsize=10,
            # fontweight="bold",
            color=val_color,
        )

    plt.title(f"Training and Validation Loss {model_name}")
    plt.xlabel("Epoch")
    if model_name == Model.DeepDenoiser.value:
        plt.ylabel("BCE Loss")
    elif model_name == Model.CleanUNet.value:
        if config.model.loss == "clean_unet_loss":
            plt.ylabel("M-STFT Loss")
        elif config.model.loss == "mae":
            plt.ylabel("L1 Loss")
        else:
            raise NotImplementedError
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(
        output_dir / f"training_validation_loss_{model_name}.png", bbox_inches="tight"
    )
    plt.show()

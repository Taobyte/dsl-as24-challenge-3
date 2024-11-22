import pathlib
import hydra
import omegaconf

import pandas as pd
import matplotlib.pyplot as plt

from models.DeepDenoiser.validate import visualize_predictions_deep_denoiser
from models.WaveDecompNet.validate import visualize_predictions_wave_decomp_net
from models.ColdDiffusion.validate import visualize_predictions_cold_diffusion
from models.CleanUNet.validate import visualize_predictions_clean_unet


def visualize_predictions(cfg: omegaconf.DictConfig):
    
    if cfg.model.model_name == "DeepDenoiser":
        visualize_predictions_deep_denoiser(cfg.user.model_path, cfg.user.data.signal_path, cfg.user.data.noise_path, cfg.model.signal_length, cfg.plot.n_examples, cfg.snrs)
    elif cfg.model.model_name == "WaveDecompNet":
        visualize_predictions_wave_decomp_net(cfg.user.model_path, cfg.user.data.signal_path, cfg.user.data.noise_path, cfg.model.signal_length, cfg.plot.n_examples, cfg.snrs)
    elif cfg.model.model_name == "ColdDiffusion":
        visualize_predictions_cold_diffusion(cfg)
    elif cfg.model.model_name == "CleanUNet":
        visualize_predictions_clean_unet(cfg.user.model_path, cfg.user.data.signal_path, cfg.user.data.noise_path, cfg.model.signal_length, cfg.plot.n_examples, cfg.snrs)
    else:
        raise ValueError(f"{cfg.model.model_name} not visualization function or not implemented")


def compare_model_and_baselines(df1_path: str, df2_path: str, df3_path: str, label1: str, label2: str, label3: str):

    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df3 = pd.read_csv(df3_path)

    print(label1)
    print(label2)
    print(label3)

    # Create a figure and axes for 3 subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Define the metrics and their labels
    metrics = [
        ('cc_mean', 'cc_std', 'Cross Correlation (CC)'),
        ('max_amp_diff_mad', 'max_amp_diff_std', 'Max Amplitude Diff (MAD)'),
        ('p_wave_mean', 'p_wave_std', 'P Wave Onset')
    ]

    # Plot each metric
    for i, (ax, (mean_col, std_col, ylabel)) in enumerate(zip(axs, metrics)):
        # Model
        ax.errorbar(df1['snr'], df1[mean_col], yerr=df1[std_col], fmt='o-', capsize=5, label=label1, 
                    color='cornflowerblue', ecolor='lightsteelblue', elinewidth=2, capthick=2, markersize=8)
        # Butterworth
        ax.errorbar(df2['snr'], df2[mean_col], yerr=df2[std_col], fmt='s-', capsize=5, label=label2, 
                    color='olivedrab', ecolor='yellowgreen', elinewidth=2, capthick=2, markersize=8)
        # DeepDenoiser
        ax.errorbar(df3['snr'], df3[mean_col], yerr=df3[std_col], fmt='^-', capsize=5, label=label3, 
                    color='tomato', ecolor='salmon', elinewidth=2, capthick=2, markersize=8)
        
        ax.set_xlabel('Signal to Noise Ratio (SNR)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        if i == 1:
            ax.set_ylim(0, 1000)

    # Set the title for the entire figure
    fig.suptitle('Comparison of Denoising Models', fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'metrics.jpg')
    plt.show()






    
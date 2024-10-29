import os

import pandas as pd
import matplotlib.pyplot as plt



def compare_two(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    # Create a figure and axes for 3 subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Define the metrics and their labels
    metrics = [
        ('cc_mean', 'cc_std', 'Cross Correlation (CC)'),
        ('max_amp_diff_mad', 'max_amp_diff_std', 'Max Amplitude Diff (MAD)'),
        ('p_wave_mean', 'p_wave_std', 'P Wave Onset')
    ]

    # Plot each metric
    for ax, (mean_col, std_col, ylabel) in zip(axs, metrics):
        ax.errorbar(df1['snr'], df1[mean_col], yerr=df1[std_col], fmt='o-', capsize=5, label=label1, 
                    color='cornflowerblue', ecolor='lightsteelblue', elinewidth=2, capthick=2, markersize=8)
        ax.errorbar(df2['snr'], df2[mean_col], yerr=df2[std_col], fmt='s-', capsize=5, label=label2, 
                    color='olivedrab', ecolor='yellowgreen', elinewidth=2, capthick=2, markersize=8)
        
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

    # Set the title for the entire figure
    fig.suptitle('Comparison of Denoising Models', fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
import os

import pandas as pd
import matplotlib.pyplot as plt



def compare_two(df1:pd.DataFrame, df2:pd.DataFrame, label1:str, label2:str):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the first model with error bars
    ax.errorbar(df1['snr'], df1['cc_mean'], yerr=df1['cc_std'], fmt='o-', capsize=5, label=label1, 
                color='cornflowerblue', ecolor='lightsteelblue', elinewidth=2, capthick=2, markersize=8)

    # Plot the second model with error bars
    ax.errorbar(df2['snr'], df2['cc_mean'], yerr=df2['cc_std'], fmt='s-', capsize=6, label=label2, 
                color='olivedrab', ecolor='yellowgreen', elinewidth=3, capthick=3, markersize=8)

    # Set the title and labels
    ax.set_title('Comparison of Denoising Models', fontsize=16)
    ax.set_xlabel('Signal to Noise Ratio (SNR)', fontsize=14)
    ax.set_ylabel('Cross Correlation (CC)', fontsize=14)

    # Add a grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend
    ax.legend(fontsize=12)

    # Set the aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()
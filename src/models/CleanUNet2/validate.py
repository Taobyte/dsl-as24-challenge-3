import omegaconf 
import hydra
import pathlib
import os
from typing import Union

import torch 
import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import Mode
from src.metrics import cross_correlation, max_amplitude_difference, p_wave_onset_difference, cross_correlation_torch, max_amplitude_difference_torch, p_wave_onset_difference_torch
from src.models.CleanUNet.dataset import CleanUNetDataset
from src.models.CleanUNet2.clean_specnet import CleanSpecNet

def visualize_predictions_clean_specnet(model, signal_path: str, noise_path:str, signal_length: int, n_examples: int, snrs:list[int], channel:int = 0, epoch="", cfg=None) -> None:
    
    print("Visualizing predictions")
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    epoch_dir = os.path.join(output_dir, str(epoch))
    os.makedirs(epoch_dir, exist_ok=True)

    if isinstance(model, str):
        if not cfg.model.train_pytorch:
            model = keras.saving.load_model(model)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = CleanSpecNet(
                channels_input=3,
                channels_output=3,
                channels_H=cfg.model.channels_H,
                encoder_n_layers=cfg.model.encoder_n_layers,
                tsfm_n_layers=cfg.model.tsfm_n_layers,
                tsfm_n_head=cfg.model.tsfm_n_head,
                tsfm_d_model=cfg.model.tsfm_d_model,
                tsfm_d_inner=cfg.model.tsfm_d_inner,
            ).to(device)

            checkpoint = torch.load(cfg.user.model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

    data_format = "channel_first" if cfg.model.train_pytorch else "channel_last"

    for snr in tqdm(snrs, total=len(snrs)):
        
        test_dl = torch.utils.data.DataLoader(CleanUNetDataset(signal_path + "/validation", noise_path + "/validation", signal_length, snr, snr, data_format=data_format, spectogram=True), batch_size=n_examples)
        input, ground_truth = next(iter(test_dl))
        predictions = model(input.float())

        print(input.shape)
        print(ground_truth.shape)

        _, axs = plt.subplots(n_examples, 3,  figsize=(15, n_examples * 3))
        time = range(signal_length)

        for i in tqdm(range(n_examples), total=n_examples):


            if not cfg.model.train_pytorch:
                axs[i,0].plot(time, input[i,:,channel]) # noisy earthquake
                axs[i,1].plot(time, ground_truth[i,:,channel]) # ground truth noise
                axs[i,2].plot(time, predictions[i,:,channel]) # predicted noise
            else:
                axs[i,0].plot(time, input[i,channel,:]) # noisy earthquake
                axs[i,1].imshow(ground_truth[i,channel,:, :], aspect='auto', origin='lower', cmap='viridis') # ground truth noise
                axs[i,2].imshow(predictions.detach().numpy()[i,channel,:, :], aspect='auto', origin='lower', cmap='viridis') # predicted noise

            
            axs[i, 0].set_ylim(-2, 2)
        
        column_titles = ["Noisy Earthquake", "Ground Truth Signal", "Prediction"]
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(epoch_dir + f'/visualization_snr_{snr}.png')
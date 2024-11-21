import omegaconf 
import hydra
import pathlib
import os
from typing import Union

import torch 
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import Mode
from src.metrics import cross_correlation, max_amplitude_difference, p_wave_onset_difference
from src.models.CleanUNet.dataset import CleanUNetDataset
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch


def get_metrics_clean_unet(
    model: keras.Model, cfg: omegaconf.DictConfig, snr: int, idx: int = 0
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = CleanUNetDataset(
        cfg.user.data.signal_path + "/validation/",
        cfg.user.data.noise_path + "/validation/",
        cfg.model.signal_length,
        snr_lower=snr,
        snr_upper=snr,
        mode=Mode.TEST,
        data_format="channel_first" if cfg.model.train_pytorch else "channel_last"
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset, cfg.model.batch_size, shuffle=False
    )
    ccs = []
    amplitudes = []
    onsets = []
    for noisy_batch, eq_batch, shifts in tqdm(test_dl, total=len(test_dl)):

        eq_batch = eq_batch.float().to(device)
        noisy_batch = noisy_batch.float().to(device)

        # get predictions
        prediction = model(noisy_batch)
        if cfg.model.train_pytorch:
            prediction = prediction.detach().cpu().numpy()
        # compute metrics
        eq_batch = eq_batch.cpu().numpy()
        prediction = np.array(prediction)
        corr = [
            cross_correlation(a, b)
            for a, b in zip(eq_batch[:, idx, :], prediction[:, idx, :])
        ]
        ccs.extend(corr)
        max_amplitude_differences = [
            max_amplitude_difference(a, b)
            for a, b in zip(eq_batch[:, idx, :], prediction[:, idx, :])
        ]
        amplitudes.extend(max_amplitude_differences)
        onset = [
            p_wave_onset_difference(a, b, shift)
            for a, b, shift in zip(
                eq_batch[:, idx, :], prediction[:, idx, :], shifts
            )
        ]
        onsets.extend(onset)

    return np.array(ccs), np.array(amplitudes), np.array(onsets)



def visualize_predictions_clean_unet(model: Union[str, keras.Model], signal_path: str, noise_path:str, signal_length: int, n_examples: int, snrs:list[int], channel:int = 0, epoch="", cfg=None) -> None:
    
    print("Visualizing predictions")
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    epoch_dir = os.path.join(output_dir, str(epoch))
    os.makedirs(epoch_dir, exist_ok=True)

    if isinstance(model, str):
        if not cfg.model.train_pytorch:
            model = keras.saving.load_model(model)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = CleanUNetPytorch(
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
        
        test_dl = torch.utils.data.DataLoader(CleanUNetDataset(signal_path + "/validation", noise_path + "/validation", signal_length, snr, snr, data_format=data_format), batch_size=n_examples)
        input, ground_truth = next(iter(test_dl))
        predictions = model(input.float())

        _, axs = plt.subplots(n_examples, 3,  figsize=(15, n_examples * 3))
        time = range(signal_length)

        for i in tqdm(range(n_examples), total=n_examples):

            if not cfg.model.train_pytorch:
                axs[i,0].plot(time, input[i,:,channel]) # noisy earthquake
                axs[i,1].plot(time, ground_truth[i,:,channel]) # ground truth noise
                axs[i,2].plot(time, predictions[i,:,channel]) # predicted noise
            else:
                axs[i,0].plot(time, input[i,channel,:]) # noisy earthquake
                axs[i,1].plot(time, ground_truth[i,channel,:]) # ground truth noise
                axs[i,2].plot(time, predictions.detach().numpy()[i,channel,:]) # predicted noise

            for j in range(3):
                axs[i, j].set_ylim(-2, 2)
        
        column_titles = ["Noisy Earthquake", "Ground Truth Signal", "Prediction"]
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(epoch_dir + f'/visualization_snr_{snr}.png')
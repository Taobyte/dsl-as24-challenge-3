import os
import glob
import random

import keras
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import Mode, Model


def get_dataloaders(signal_path: str, noise_path: str, shuffle = False, model = Model.DeepDenoiser, batch_size=32) -> tuple[DataLoader, DataLoader]:

    if model == Model.DeepDenoiser:

        train_assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.TRAIN)
        val_assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.VALIDATION)
        
        train_dataset = CombinedDeepDenoiserDataset(InputSignals(train_assoc), EventMasks(train_assoc))
        val_dataset = CombinedDeepDenoiserDataset(InputSignals(val_assoc), EventMasks(val_assoc))
    

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dl, validation_dl


def get_signal_noise_assoc(signal_path: str, noise_path: str, mode: Mode, size_testset = 1000, snr=None) -> list[tuple[str, str, float, int]]:
    """generates a signal to noise file association from folders
    Allows defining a standard data association for reproducibility. 
    Args:
        - signal_path: path to the signal folder
        - noise_path: path to the noise_folder
        - train: boolean (use training set)
        - mode: enum TRAIN, VALIDATION, TEST 
        - size_testset: integer for fixing testset size
        - snr: float for fixing the signal to noise ratio
    Returns:
        - a list of tuples, where each tuple contains:
          (signal file, noise file, random SNR, random event shift)

    """

    if mode == Mode.TRAIN:

        signal_files = glob.glob(f"{signal_path}/train/**/*.npz", recursive=True)
        noise_files = glob.glob(f"{noise_path}/train/**/*.npz", recursive=True)

    elif mode == Mode.VALIDATION:

        signal_files = glob.glob(f"{signal_path}/validation/**/*.npz", recursive=True)[:-size_testset]
        noise_files = glob.glob(f"{noise_path}/validation/**/*.npz", recursive=True)[:-size_testset]
    
    elif mode == Mode.TEST:

        signal_files = glob.glob(f"{signal_path}/validation/**/*.npz", recursive=True)[-size_testset:]
        noise_files = glob.glob(f"{noise_path}/validation/**/*.npz", recursive=True)[-size_testset:]
    
    else:

        print(f"Mode {mode} not supported!")
        
    # shuffle
    random.shuffle(signal_files)
    random.shuffle(noise_files)

    assoc = []
    for i in range(len(signal_files)):
        n = np.random.randint(0, len(noise_files))
        snr_random = snr if snr else np.random.uniform(0.2,1.5)
        event_shift = np.random.randint(1000,6000)
        assoc.append((signal_files[i], noise_files[n], snr_random, event_shift))

    return assoc


class InputSignals(Dataset):

    def __init__(self, signal_noise_association:list, mode = Mode.TRAIN, snr = 1.0):
        """
        Args:
            signal_noise_association: a list containing tuples for signal and noise filenames
        """

        self.signal_noise_assoc = signal_noise_association
        self.signal_length = 6120
        self.mode = mode

    def __len__(self) -> int:
        return len(self.signal_noise_assoc)

    def __getitem__(self, idx: int) -> th.Tensor:

        eq = np.load(self.signal_noise_assoc[idx][0], allow_pickle=True)
        noise = np.load(self.signal_noise_assoc[idx][1], allow_pickle=True)
        snr_random = self.signal_noise_assoc[idx][2]
        event_shift = self.signal_noise_assoc[idx][3]
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + self.signal_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + self.signal_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + self.signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["noise_waveform_Z"][:self.signal_length]
        N_noise = noise["noise_waveform_N"][:self.signal_length]
        E_noise = noise["noise_waveform_E"][:self.signal_length]
        
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        if self.mode == Mode.TRAIN:
            ratio = snr_random
        elif self.mode == Mode.TEST:
            ratio = self.snr
        else: 
            print(f"Not supported mode {self.mode}")
            ratio = 0

        signal_std = np.std(eq_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)  
        noise_std = np.std(noise_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)
        snr_original = signal_std / (noise_std + 1e-4)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * ratio  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        return noisy_eq


class EventMasks(Dataset):

    def __init__(self, signal_noise_association:list):
        """
        Args:
            signal_noise_association: a list containing tuples for signal and noise filenames
        """

        self.signal_noise_assoc = signal_noise_association
        self.signal_length = 6120

        # STFT parameters
        self.frame_length = 100
        self.frame_step = 24
        self.fft_size = 126

    def __len__(self) -> int:
        return len(self.signal_noise_assoc)

    def __getitem__(self, idx) -> th.Tensor:

        eq = np.load(self.signal_noise_assoc[idx][0], allow_pickle=True)
        noise = np.load(self.signal_noise_assoc[idx][1], allow_pickle=True)
        snr_random = self.signal_noise_assoc[idx][2]
        event_shift = self.signal_noise_assoc[idx][3]
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + self.signal_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + self.signal_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + self.signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["noise_waveform_Z"][:self.signal_length]
        N_noise = noise["noise_waveform_N"][:self.signal_length]
        E_noise = noise["noise_waveform_E"][:self.signal_length]

        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        signal_std = np.std(eq_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)  
        noise_std = np.std(noise_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)
        snr_original = signal_std / (noise_std + 1e-4)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR

        stft = keras.ops.stft(eq_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_eq = np.concatenate([stft[0],stft[1]], axis=0)

        stft = keras.ops.stft(noise_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_noise = np.concatenate([stft[0],stft[1]], axis=0)

        mask = np.abs(stft_eq) / (np.abs(stft_noise) + np.abs(stft_eq) + 1e-4)
        
        return mask
        

class CombinedDeepDenoiserDataset(Dataset):
    def __init__(self, input_signals: InputSignals, event_masks: EventMasks):
        """
        Args:
            input_signals: Instance of the InputSignals dataset.
            event_masks: Instance of the EventMasks dataset.
        """
        self.input_signals = input_signals
        self.event_masks = event_masks
        assert len(self.input_signals) == len(self.event_masks), "Datasets must be of equal length."

    def __len__(self):
        return len(self.input_signals)

    def __getitem__(self, idx):
        # Fetch the noisy earthquake signal (input)
        noisy_eq = self.input_signals[idx]
        
        # Fetch the corresponding mask (ground truth)
        mask = self.event_masks[idx]
        
        return noisy_eq, mask

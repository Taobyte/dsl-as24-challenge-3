import glob
import random

import keras
import torch as th
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from numpy import ndarray

from src.utils import Mode, Model


def get_dataloaders(signal_path: str, noise_path: str, shuffle = False, batch_size=32) -> tuple[DataLoader, DataLoader]:

    train_assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.TRAIN)
    val_assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.VALIDATION)
    
    train_dataset = CombinedDeepDenoiserDataset(train_assoc)
    val_dataset = CombinedDeepDenoiserDataset(val_assoc)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dl, validation_dl


def get_signal_noise_assoc(signal_path: str, noise_path: str, mode: Mode, size_testset = 1000, 
                           snr=lambda : np.random.uniform(0.1,1.1)
                           ) -> list[tuple[str, str, float, int]]:
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
    print(mode)

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

        assert False, f"Mode {mode} not supported!"
        
    # shuffle
    random.shuffle(signal_files)
    random.shuffle(noise_files)

    assoc = []
    for i in range(len(signal_files)):
        n = np.random.randint(0, len(noise_files))
        snr_random = snr()
        event_shift = np.random.randint(1000,6000)
        assoc.append((signal_files[i], noise_files[n], snr_random, event_shift))

    return assoc

def load_traces_and_shift(assoc: list, trace_length: int) -> tuple[ndarray, ndarray, ndarray]:

    eq_traces = []
    noise_traces = []
    shifts = []

    for eq_path, noise_path, _, event_shift in assoc:

        eq = np.load(eq_path, allow_pickle=True)
        noise = np.load(noise_path, allow_pickle=True)

        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + trace_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + trace_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + trace_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["noise_waveform_Z"][:trace_length]
        N_noise = noise["noise_waveform_N"][:trace_length]
        E_noise = noise["noise_waveform_E"][:trace_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        eq_traces.append(eq_stacked)
        noise_traces.append(noise_stacked)
        shifts.append(event_shift)

    eq_traces = np.array(eq_traces)
    noise_traces = np.array(noise_traces)
    shifts = np.array(shifts)

    return eq_traces, noise_traces, shifts


class InputSignals(Dataset):

    def __init__(self, signal_noise_association:list, mode = Mode.TRAIN, snr = 1.0):
        """
        Args:
            signal_noise_association: a list containing tuples for signal and noise filenames
        """

        self.signal_noise_assoc = signal_noise_association
        self.signal_length = 6120
        self.mode = mode
        self.snr = snr

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
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * ratio  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        if self.mode == Mode.TRAIN:
            return noisy_eq
        elif self.mode == Mode.TEST:
            return noisy_eq, eq_stacked, event_shift


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
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR

        stft = keras.ops.stft(eq_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_eq = np.concatenate([stft[0],stft[1]], axis=0)

        stft = keras.ops.stft(noise_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_noise = np.concatenate([stft[0],stft[1]], axis=0)

        mask = np.abs(stft_eq) / (np.abs(stft_noise) + np.abs(stft_eq) + 1e-10)
        
        return mask
        

class CombinedDeepDenoiserDataset(Dataset):
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
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        stft = keras.ops.stft(eq_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_eq = np.concatenate([stft[0],stft[1]], axis=0)

        stft = keras.ops.stft(noise_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_noise = np.concatenate([stft[0],stft[1]], axis=0)

        mask = np.abs(stft_eq) / (np.abs(stft_noise) + np.abs(stft_eq) + 1e-10)
        
        return noisy_eq, mask
    


class CSVDataset(Dataset):

    def __init__(self, path: str, signal_length: int, snr_lower:float, snr_upper:float, mode: Mode):
        
        print("start loading pickle files")
        
        if mode == Mode.TRAIN:
            self.signal_df = pd.read_pickle(path + '/signal_train.pkl')
            self.noise_df = pd.read_pickle(path + '/noise_train.pkl')
        elif mode == Mode.VALIDATION:
            self.signal_df = pd.read_pickle(path + '/signal_validation.pkl')
            self.noise_df = pd.read_pickle(path + '/noise_validation.pkl')
        else:
            assert False, f'mode {mode} not implemented yet'
        
        print("finished loading pickle files")

        self.signal_length = signal_length
        self.snr_lower = snr_lower
        self.snr_upper = snr_upper

        # STFT parameters
        self.frame_length = 100
        self.frame_step = 24
        self.fft_size = 126
    
    def __len__(self) -> int:
        return len(self.signal_df)
    
    def __getitem__(self, idx) -> tuple[ndarray, ndarray]:
        
        eq = self.signal_df.iloc[idx]
        random_noise_idx = np.random.randint(len(self.noise_df))
        noise = self.noise_df.iloc[random_noise_idx]
        assert self.snr_lower <= self.snr_upper
        snr_random = np.random.uniform(self.snr_lower,self.snr_upper)
        event_shift = np.random.randint(1000,6000)
        
        Z_eq = eq["Z"][event_shift : event_shift + self.signal_length]
        N_eq = eq["N"][event_shift : event_shift + self.signal_length]
        E_eq = eq["E"][event_shift : event_shift + self.signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["Z"][:self.signal_length]
        N_noise = noise["N"][:self.signal_length]
        E_noise = noise["E"][:self.signal_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        signal_std = np.std(eq_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)  
        noise_std = np.std(noise_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        stft = keras.ops.stft(eq_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_eq = np.concatenate([stft[0],stft[1]], axis=0)

        stft = keras.ops.stft(noise_stacked, self.frame_length, self.frame_step, self.fft_size)
        stft_noise = np.concatenate([stft[0],stft[1]], axis=0)

        mask = np.abs(stft_eq) / (np.abs(stft_noise) + np.abs(stft_eq) + 1e-10)
        
        return noisy_eq, mask

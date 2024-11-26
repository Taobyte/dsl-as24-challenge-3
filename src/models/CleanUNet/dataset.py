import glob
import omegaconf

import numpy as np
from numpy import ndarray
import torch
import pandas as pd
import einops

from src.utils import Mode
from src.models.CleanUNet.stft_loss import stft

class CleanUNetDataset(torch.utils.data.Dataset):

    def __init__(self, signal_path: str, noise_path: str, signal_length: int, snr_lower: int=0.1, snr_upper: int=2.0, mode: Mode=Mode.TRAIN, random=True, data_format="channel_last", spectogram=False, fft_size=126, hop_size=24, win_length=100, window="hann_window"):
        
        self.signal_files = glob.glob(f"{signal_path}/**/*.npz", recursive=True) if random else sorted(glob.glob(f"{signal_path}/**/*.npz", recursive=True))
        self.noise_files = glob.glob(f"{noise_path}/**/*.npz", recursive=True) if random else sorted(glob.glob(f"{noise_path}/**/*.npz", recursive=True))

        print(f'length signal files: {len(self.signal_files)}')
        print(f'length noise files: {len(self.noise_files)}')

        self.signal_length = signal_length
        
        self.snr_lower = snr_lower
        self.snr_upper = snr_upper

        self.mode = mode
        self.data_format = data_format
        self.spectogram = spectogram
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

        self.random = random
        if not random:
            np.random.seed(123)

    def __len__(self) -> int:
        return len(self.signal_files)
    
    def __getitem__(self, idx) -> tuple[ndarray, ndarray]:

        
        eq = np.load(self.signal_files[idx], allow_pickle=True)
        random_noise_idx = np.random.randint(len(self.noise_files)) if self.random else idx
        noise = np.load(self.noise_files[random_noise_idx], allow_pickle=True)
        assert self.snr_lower <= self.snr_upper
        snr_random = np.random.uniform(self.snr_lower,self.snr_upper) if self.random else 1.0
        event_shift = np.random.randint(6000-self.signal_length + 500,6000) if self.random else 3000
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + self.signal_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + self.signal_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + self.signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=1)

        Z_noise = noise["noise_waveform_Z"][:self.signal_length]
        N_noise = noise["noise_waveform_N"][:self.signal_length]
        E_noise = noise["noise_waveform_E"][:self.signal_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=1)

        max_val = max(np.max(np.abs(noise_stacked)), np.max(np.abs(eq_stacked))) + 1e-10
        eq_stacked = eq_stacked / max_val
        noise_stacked = noise_stacked / max_val

        signal_std = np.std(eq_stacked[6000-event_shift:6500-event_shift, :], axis=0) 
        noise_std = np.std(noise_stacked[6000-event_shift:6500-event_shift,:], axis=0)        
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        if self.data_format == "channel_first":
            noisy_eq = einops.rearrange(noisy_eq, "t c -> c t")
            eq_stacked = einops.rearrange(eq_stacked, "t c -> c t")
            eq_stacked = torch.from_numpy(eq_stacked)
            noisy_eq = torch.from_numpy(noisy_eq)
        
        if self.spectogram:
            assert self.data_format == "channel_first"
            eq_stacked = stft(eq_stacked, self.fft_size, self.hop_size, self.win_length, self.window) # (3*B, #frames, fft_size // 2 + 1)
            eq_stacked = einops.rearrange(eq_stacked, "repeat t c -> repeat c t", repeat=3)

        if self.mode == Mode.TEST:
            return noisy_eq, eq_stacked, event_shift
        else:
            return noisy_eq, eq_stacked


class CleanUNetDatasetCSV(torch.utils.data.Dataset):

    def __init__(self, path: str, signal_length: int, snr_lower: int=0.1, snr_upper: int=2.0, mode: Mode=Mode.TRAIN, data_format="channel_last", spectogram=False, fft_size=126, hop_size=24, win_length=100, window="hann_window"):
        
        print("start loading pickle files")
        
        if mode == Mode.TRAIN:
            self.signal_df = pd.read_pickle(path + '/signal_train.pkl')
            self.noise_df = pd.read_pickle(path + '/noise_train.pkl')
        elif mode == Mode.VALIDATION:
            self.signal_df = pd.read_pickle(path + '/signal_validation.pkl')
            self.noise_df = pd.read_pickle(path + '/noise_validation.pkl')
        elif mode == Mode.TEST:
            self.signal_df = pd.read_pickle(path + '/signal_validation.pkl')
            self.noise_df = pd.read_pickle(path + '/noise_validation.pkl')
        else:
            assert False, f'mode {mode} not implemented yet'
        
        print("finished loading pickle files")

        self.signal_length = signal_length
        
        self.snr_lower = snr_lower
        self.snr_upper = snr_upper

        self.mode = mode
        self.data_format = data_format
        self.spectogram = spectogram
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def __len__(self) -> int:
        return len(self.signal_df)
    
    def __getitem__(self, idx) -> tuple[ndarray, ndarray]:
        
        eq = self.signal_df.iloc[idx]
        random_noise_idx = np.random.randint(len(self.noise_df))
        noise = self.noise_df.iloc[random_noise_idx]
        assert self.snr_lower <= self.snr_upper
        snr_random = np.random.uniform(self.snr_lower,self.snr_upper)
        event_shift = np.random.randint(6000-self.signal_length + 500,6000)
        
        Z_eq = eq["Z"][event_shift : event_shift + self.signal_length]
        N_eq = eq["N"][event_shift : event_shift + self.signal_length]
        E_eq = eq["E"][event_shift : event_shift + self.signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=1)

        Z_noise = noise["Z"][:self.signal_length]
        N_noise = noise["N"][:self.signal_length]
        E_noise = noise["E"][:self.signal_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=1)

        max_val = max(np.max(np.abs(noise_stacked)), np.max(np.abs(eq_stacked))) + 1e-10
        eq_stacked = eq_stacked / max_val
        noise_stacked = noise_stacked / max_val

        signal_std = np.std(eq_stacked[6000-event_shift:6500-event_shift, :], axis=0) 
        noise_std = np.std(noise_stacked[6000-event_shift:6500-event_shift,:], axis=0)        
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
        noisy_eq = eq_stacked + noise_stacked # recombine

        # channel_first means we use the Pytorch model instead of the keras model
        if self.data_format == "channel_first":
            noisy_eq = einops.rearrange(noisy_eq, "t c -> c t")
            eq_stacked = einops.rearrange(eq_stacked, "t c -> c t")
            # numpy array to torch tensor 
            eq_stacked = torch.from_numpy(eq_stacked)
            noisy_eq = torch.from_numpy(noisy_eq)
        
        if self.spectogram:
            assert self.data_format == "channel_first"
            # eq_stacked = einops.rearrange(eq_stacked, "c t -> (b c) t") 
            eq_stacked = stft(eq_stacked, self.fft_size, self.hop_size, self.win_length, self.window) # (3*B, #frames, fft_size // 2 + 1)
            eq_stacked = einops.rearrange(eq_stacked, "repeat t c -> repeat c t", repeat=3)

        if self.mode == Mode.TEST:
            return noisy_eq, eq_stacked, event_shift
        else:
            return noisy_eq, eq_stacked
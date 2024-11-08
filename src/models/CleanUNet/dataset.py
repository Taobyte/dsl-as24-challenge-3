import glob

import numpy as np
from numpy import ndarray
import torch

from src.utils import Mode


class CleanUNetDataset(torch.utils.data.Dataset):

    def __init__(self, signal_path: str, noise_path: str, signal_length: int, snr_lower: int=0.1, snr_upper: int=2.0, event_shift_start:int=1000, mode: Mode=Mode.TRAIN):
        
        self.signal_files = glob.glob(f"{signal_path}/**/*.npz", recursive=True)
        self.noise_files = glob.glob(f"{noise_path}/**/*.npz", recursive=True)

        print(f'length signal files: {len(self.signal_files)}')
        print(f'length noise files: {len(self.noise_files)}')

        self.signal_length = signal_length
        
        self.snr_lower = snr_lower
        self.snr_upper = snr_upper

        assert event_shift_start >= 0 and event_shift_start <= 6000
        self.event_shift_start = event_shift_start

        self.mode = mode

    
    def __len__(self) -> int:
        return len(self.signal_files)
    
    def __getitem__(self, idx) -> tuple[ndarray, ndarray]:
        
        eq = np.load(self.signal_files[idx], allow_pickle=True)
        random_noise_idx = np.random.randint(len(self.noise_files))
        noise = np.load(self.noise_files[random_noise_idx], allow_pickle=True)
        assert self.snr_lower <= self.snr_upper
        snr_random = np.random.uniform(self.snr_lower,self.snr_upper)
        event_shift = np.random.randint(6000-self.signal_length + 500,6000)
        
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

        if self.mode == Mode.TEST:
            return noisy_eq, eq_stacked, event_shift
        else:
            return noisy_eq, eq_stacked
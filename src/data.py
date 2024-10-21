import os
import glob
import random

import keras
import scipy
import pandas as pd
import torch as th
import numpy as np
from numpy import ndarray
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader, Subset


def get_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:

    signal_train_path = args.signal_path + "/train"
    signal_validation_path = args.signal_path + "/validation"

    noise_train_path = args.noise_path + "/train"
    noise_validation_path = args.noise_path + "/validation"

    if args.deepdenoiser:

        train_dataset = DeepDenoiserDataset(signal_train_path, noise_train_path)
        validation_dataset = DeepDenoiserDataset(
            signal_validation_path, noise_validation_path
        )
    elif args.colddiffusion:

        train_dataset = ColdDiffusionDataset(signal_train_path, noise_train_path)
        validation_dataset = ColdDiffusionDataset(
            signal_validation_path, noise_validation_path
        )


    if args.dataset_length:
        train_indices = th.randint(len(train_dataset), (args.dataset_length,))
        validation_indices = th.randint(len(validation_dataset), (args.dataset_length,))
        train_dataset = Subset(train_dataset, train_indices)
        validation_dataset = Subset(validation_dataset, validation_indices)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dl = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dl, validation_dl


class DeepDenoiserDataset(Dataset):

    def __init__(self, signal_folder_path: str, noise_folder_path: str):
        """
        Args:
            signal_folder_path (str): Path to earthquake signal folder containing .npz files.
            noise_folder_path (str): Path to noise folder containing .npz files.
        """

        self.signal_folder_path = signal_folder_path
        self.noise_folder_path = noise_folder_path
        self.eq_signal_files = glob.glob(
            f"{signal_folder_path}/**/*.npz", recursive=True
        )
        self.noise_files = glob.glob(f"{noise_folder_path}/**/*.npz", recursive=True)
        self.signal_length = 3000

        # scipy hyperparameters
        self.fs = 100
        self.nperseg = 30
        self.nfft = 60

        # sample hyperparameters
        self.noise_mean = 2
        self.noise_std = 1

    def __len__(self) -> int:
        return len(self.eq_signal_files)

    def __getitem__(self, idx) -> th.Tensor:

        while True:

            eq_path = self.eq_signal_files[idx]
            eq = np.load(eq_path, allow_pickle=True)
            eq_name = (os.path.splitext(os.path.basename(eq_path)))[0]

            noise_to_small = True
            while noise_to_small:
                noise_idx = np.random.randint(0, len(self.noise_files))
                noise_path = self.noise_files[noise_idx]
                noise = np.load(noise_path, allow_pickle=True)
                if len(noise["noise_waveform_Z"]) >= self.signal_length:
                    noise_to_small = False

            noise_name = (os.path.splitext(os.path.basename(noise_path)))[0]

            noise_seq_len = len(noise["noise_waveform_Z"])
            assert noise_seq_len >= self.signal_length

            eq_start = np.random.randint(low=3000, high=6000)
            noise_start = np.random.randint(
                low=0, high=max(noise_seq_len - self.signal_length, 1)
            )

            Z_eq = eq["earthquake_waveform_Z"][eq_start : eq_start + self.signal_length]
            N_eq = eq["earthquake_waveform_N"][eq_start : eq_start + self.signal_length]
            E_eq = eq["earthquake_waveform_E"][eq_start : eq_start + self.signal_length]
            eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)
            eq_tensor = th.from_numpy(eq_stacked)

            Z_noise = noise["noise_waveform_Z"][
                noise_start : noise_start + self.signal_length
            ]
            N_noise = noise["noise_waveform_N"][
                noise_start : noise_start + self.signal_length
            ]
            E_noise = noise["noise_waveform_E"][
                noise_start : noise_start + self.signal_length
            ]

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)
            noise_tensor = th.from_numpy(noise_stacked)

            # sample random channel
            j = np.random.choice([0, 1, 2])

            eq = eq_stacked[j]
            noise = noise_stacked[j]

            def compute_stft(signal: ndarray) -> ndarray:

                f, t, transform = scipy.signal.stft(
                    signal,
                    fs=self.fs,
                    nperseg=self.nperseg,
                    nfft=self.nfft,
                    boundary="zeros",
                )

                return transform

            stft_eq = compute_stft(eq)
            stft_noise = compute_stft(noise)

            if np.isinf(stft_eq).any() or np.isnan(stft_eq).any():
                continue

            if np.isinf(stft_noise).any() or np.isnan(stft_noise).any():
                continue

            if np.random.random() < 0.9 and np.std(stft_eq) > 0.001:

                stft_eq = stft_eq / np.std(stft_eq)

                if np.isinf(stft_eq).any() or np.isnan(stft_eq).any():
                    continue

                if np.isinf(stft_noise).any() or np.isnan(stft_noise).any():
                    continue

                if np.random.random() < 0.2:
                    stft_eq = np.fliplr(stft_eq)

            ratio = 0
            while ratio <= 0:
                ratio = self.noise_mean + np.random.randn() * self.noise_std

            noisy = stft_eq + ratio * stft_noise
            noisy = np.stack([noisy.real, noisy.imag], axis=-1)

            if np.isnan(noisy).any() or np.isinf(noisy).any():
                continue

            noisy = noisy / np.std(noisy)
            tmp_mask = np.abs(stft_eq) / (
                np.abs(stft_eq) + np.abs(ratio * stft_noise) + 1e-4
            )
            tmp_mask[tmp_mask >= 1] = 1
            tmp_mask[tmp_mask <= 0] = 0
            mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], 2])
            mask[:, :, 0] = tmp_mask
            mask[:, :, 1] = 1 - tmp_mask

            th_mask = th.from_numpy(mask)
            th_noisy = th.from_numpy(noisy)

            return th_noisy, th_mask


class SeismicDataset(Dataset):

    def __init__(
        self, signal_folder_path: str, noise_folder_path: str, randomized=False
    ):
        """
        Args:
            signal_folder_path (str): Path to earthquake signal folder containing .npz files.
            noise_folder_path (str): Path to noise folder containing .npz files.
        """

        self.signal_folder_path = signal_folder_path
        self.noise_folder_path = noise_folder_path
        self.eq_signal_files = glob.glob(
            f"{signal_folder_path}/**/*.npz", recursive=True
        )
        self.noise_files = glob.glob(f"{noise_folder_path}/**/*.npz", recursive=True)
        self.randomized = randomized

        self.signal_length = 6000

    def __len__(self) -> int:
        return len(self.eq_signal_files)

    def __getitem__(self, idx) -> tuple[ndarray, ndarray]:

        eq_path = self.eq_signal_files[idx]
        eq = np.load(eq_path, allow_pickle=True)
        eq_name = (os.path.splitext(os.path.basename(eq_path)))[0]

        while True:

            noise_to_small = True
            while noise_to_small:
                noise_idx = np.random.randint(0, len(self.noise_files))
                noise_path = self.noise_files[noise_idx]
                noise = np.load(noise_path, allow_pickle=True)
                if len(noise["noise_waveform_Z"]) >= self.signal_length:
                    noise_to_small = False

            noise_length = len(noise["noise_waveform_Z"])

            noise_name = (os.path.splitext(os.path.basename(noise_path)))[0]

            eq_start = 0
            noise_start = 0
            if self.randomized:
                eq_start = np.random.randint(low=0, high=6000)
                noise_start = np.random.randint(
                    low=0, high=max(noise_length - self.signal_length, 1)
                )

            Z_eq = eq["earthquake_waveform_Z"][eq_start : eq_start + self.signal_length]
            N_eq = eq["earthquake_waveform_N"][eq_start : eq_start + self.signal_length]
            E_eq = eq["earthquake_waveform_E"][eq_start : eq_start + self.signal_length]
            eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

            Z_noise = noise["noise_waveform_Z"][
                noise_start : noise_start + self.signal_length
            ]
            N_noise = noise["noise_waveform_N"][
                noise_start : noise_start + self.signal_length
            ]
            E_noise = noise["noise_waveform_E"][
                noise_start : noise_start + self.signal_length
            ]

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

            j = np.random.choice([0, 1, 2])

            p_wave_start = 6000 - eq_start

            noise = noise_stacked[j]
            noise_max = np.max(np.abs(noise))
            if noise_max > 0.00001:
                noise = noise / noise_max
            eq = eq_stacked[j]
            eq_max = np.max(np.abs(eq))
            if eq_max > 0.00001:
                eq = eq / eq_max

            noise_scaling = 0.5

            noisy_eq = eq + noise * noise_scaling

            return noisy_eq, eq


class ColdDiffusionDataset(Dataset):

    def __init__(
        self,
        path: str,
        is_noise = False
    ):

        self.eq_signal_files = glob.glob(
            f"{path}/**/*.npz", recursive=True
        )

        self.signal_length = 3000

    def __len__(self):
        return len(self.eq_signal_files)

    def __getitem__(self, idx):

        while True:

            eq_path = self.eq_signal_files[idx]
            eq = np.load(eq_path, allow_pickle=True)
            eq_name = (os.path.splitext(os.path.basename(eq_path)))[0]

            noise_to_small = True
            while noise_to_small:
                noise_idx = np.random.randint(0, len(self.noise_files))
                noise_path = self.noise_files[noise_idx]
                noise = np.load(noise_path, allow_pickle=True)
                if len(noise["noise_waveform_Z"]) >= self.signal_length:
                    noise_to_small = False

            noise_name = (os.path.splitext(os.path.basename(noise_path)))[0]

            noise_seq_len = len(noise["noise_waveform_Z"])
            assert noise_seq_len >= self.signal_length

            eq_start = np.random.randint(low=3000, high=6000)
            noise_start = np.random.randint(
                low=0, high=max(noise_seq_len - self.signal_length, 1)
            )

            Z_eq = eq["earthquake_waveform_Z"][eq_start : eq_start + self.signal_length]
            N_eq = eq["earthquake_waveform_N"][eq_start : eq_start + self.signal_length]
            E_eq = eq["earthquake_waveform_E"][eq_start : eq_start + self.signal_length]
            eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)
            eq_tensor = th.from_numpy(eq_stacked)
            eq_tensor_normalized = eq_tensor / eq_tensor.abs().max()

            Z_noise = noise["noise_waveform_Z"][
                noise_start : noise_start + self.signal_length
            ]
            N_noise = noise["noise_waveform_N"][
                noise_start : noise_start + self.signal_length
            ]
            E_noise = noise["noise_waveform_E"][
                noise_start : noise_start + self.signal_length
            ]

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)
            noise_tensor = th.from_numpy(noise_stacked)
            noise_tensor_normalized = noise_tensor / noise_tensor.abs().max()

            noise_reduction = th.randint(*self.range_rnf) * 0.01

            noisy_eq = (
                eq_tensor_normalized[self.channel_type]
                + noise_reduction * noise_tensor_normalized[self.channel_type]
            )

            return noisy_eq, eq_tensor_normalized[self.channel_type]


class NikoDataset(Dataset):

    def __init__(self, signal_folder_path: str, noise_folder_path: str, train = True):
        """
        Args:
            signal_folder_path (str): Path to earthquake signal folder containing .npz files.
            noise_folder_path (str): Path to noise folder containing .npz files.
        """
        if train:
            signal_df = pd.read_pickle(signal_folder_path + "/signal_train.pkl")
            noise_df = pd.read_pickle(noise_folder_path + "/noise_train.pkl")
        else:
            signal_df = pd.read_pickle(signal_folder_path + "/signal_validation.pkl")
            noise_df = pd.read_pickle(noise_folder_path + "/noise_validation.pkl")
        
        self.signal_df = signal_df
        self.noise_df = noise_df

        self.signal_length = 3000
    
    def __len__(self):
        return len(self.signal_df)
    
    def __getitem__(self, idx):

        random_channel = np.random.choice(['Z', 'E', 'N'])
        noise_idx = np.random.randint(0, len(self.noise_df))

        signal = self.signal_df[random_channel].iloc[idx]
        noise = self.noise_df[random_channel].iloc[noise_idx]

        signal_std = np.std(signal[6000:6500])  # compute signals std over main event signal
        noise_std = np.std(noise[6000:6500])  #  compute nosie std 
        snr_original = signal_std / noise_std

        # randomly shift event start
        event_shift = np.random.randint(1000,6000)

        # change the SNR
        noise_snr_mod = (noise * snr_original)[event_shift: event_shift + self.signal_length]  # rescale noise so that SNR=1
        snr_random = np.random.uniform(0.5,2)  # random SNR     
        event_snr_mod = (signal * snr_random)[:self.signal_length]  # rescale event to desired SNR

        def compute_stft(signal: ndarray) -> ndarray:

            f, t, transform = scipy.signal.stft(
                signal,
                fs=self.fs,
                nperseg=self.nperseg,
                nfft=self.nfft,
                boundary="zeros",
            )

            return transform

        stft_eq = compute_stft(event_snr_mod)
        stft_noise = compute_stft(noise_snr_mod)

        noisy = stft_eq + stft_noise
        noisy = np.stack([noisy.real, noisy.imag], axis=-1)

        noisy = noisy / np.std(noisy)
        tmp_mask = np.abs(stft_eq) / (
            np.abs(stft_eq) + np.abs(stft_noise) + 1e-4
        )
        tmp_mask[tmp_mask >= 1] = 1
        tmp_mask[tmp_mask <= 0] = 0
        mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], 2])
        mask[:, :, 0] = tmp_mask
        mask[:, :, 1] = 1 - tmp_mask

        th_mask = th.from_numpy(mask)
        th_noisy = th.from_numpy(noisy)    
        
        return th_noisy, th_mask


class DeepDenoiserDatasetTest(Dataset):

    def __init__(self, signal_folder_path: str, noise_folder_path: str):
        """
        Args:
            signal_folder_path (str): Path to earthquake signal folder containing .npz files.
            noise_folder_path (str): Path to noise folder containing .npz files.
        """

        self.signal_folder_path = signal_folder_path
        self.noise_folder_path = noise_folder_path
        self.eq_signal_files = glob.glob(
            f"{signal_folder_path}/**/*.npz", recursive=True
        )
        self.noise_files = glob.glob(f"{noise_folder_path}/**/*.npz", recursive=True)
        self.signal_length = 3000

        # scipy hyperparameters
        self.fs = 100
        self.nperseg = 30
        self.nfft = 60

        # sample hyperparameters
        self.noise_mean = 2
        self.noise_std = 1

    def __len__(self) -> int:
        return len(self.eq_signal_files)

    def __getitem__(self, idx) -> th.Tensor:

        while True:

            eq_path = self.eq_signal_files[idx]
            eq = np.load(eq_path, allow_pickle=True)
            eq_name = (os.path.splitext(os.path.basename(eq_path)))[0]

            noise_to_small = True
            while noise_to_small:
                noise_idx = np.random.randint(0, len(self.noise_files))
                noise_path = self.noise_files[noise_idx]
                noise = np.load(noise_path, allow_pickle=True)
                if len(noise["noise_waveform_Z"]) >= self.signal_length:
                    noise_to_small = False

            noise_name = (os.path.splitext(os.path.basename(noise_path)))[0]

            noise_seq_len = len(noise["noise_waveform_Z"])
            assert noise_seq_len >= self.signal_length

            eq_start = np.random.randint(low=3000, high=6000)
            noise_start = np.random.randint(
                low=0, high=max(noise_seq_len - self.signal_length, 1)
            )

            Z_eq = eq["earthquake_waveform_Z"][eq_start : eq_start + self.signal_length]
            N_eq = eq["earthquake_waveform_N"][eq_start : eq_start + self.signal_length]
            E_eq = eq["earthquake_waveform_E"][eq_start : eq_start + self.signal_length]
            eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)
            eq_tensor = th.from_numpy(eq_stacked)

            Z_noise = noise["noise_waveform_Z"][
                noise_start : noise_start + self.signal_length
            ]
            N_noise = noise["noise_waveform_N"][
                noise_start : noise_start + self.signal_length
            ]
            E_noise = noise["noise_waveform_E"][
                noise_start : noise_start + self.signal_length
            ]

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)
            noise_tensor = th.from_numpy(noise_stacked)

            # sample random channel
            j = np.random.choice([0, 1, 2])

            eq = eq_stacked[j]
            noise = noise_stacked[j]

            def compute_stft(signal: ndarray) -> ndarray:

                f, t, transform = scipy.signal.stft(
                    signal,
                    fs=self.fs,
                    nperseg=self.nperseg,
                    nfft=self.nfft,
                    boundary="zeros",
                )

                return transform

            stft_eq = compute_stft(eq)
            stft_noise = compute_stft(noise)

            if np.isinf(stft_eq).any() or np.isnan(stft_eq).any():
                continue

            if np.isinf(stft_noise).any() or np.isnan(stft_noise).any():
                continue

            if np.random.random() < 0.9 and np.std(stft_eq) > 0.001:

                stft_eq = stft_eq / np.std(stft_eq)

                if np.isinf(stft_eq).any() or np.isnan(stft_eq).any():
                    continue

                if np.isinf(stft_noise).any() or np.isnan(stft_noise).any():
                    continue

                if np.random.random() < 0.2:
                    stft_eq = np.fliplr(stft_eq)

            ratio = 0
            while ratio <= 0:
                ratio = self.noise_mean + np.random.randn() * self.noise_std

            noisy = stft_eq + ratio * stft_noise
            noisy = np.stack([noisy.real, noisy.imag], axis=-1)

            if np.isnan(noisy).any() or np.isinf(noisy).any():
                continue

            noisy = noisy / np.std(noisy)
            tmp_mask = np.abs(stft_eq) / (
                np.abs(stft_eq) + np.abs(ratio * stft_noise) + 1e-4
            )
            tmp_mask[tmp_mask >= 1] = 1
            tmp_mask[tmp_mask <= 0] = 0
            mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], 2])
            mask[:, :, 0] = tmp_mask
            mask[:, :, 1] = 1 - tmp_mask

            th_mask = th.from_numpy(mask)
            th_noisy = th.from_numpy(noisy)

            return th_noisy, th_mask, eq, noise


def get_signal_noise_assoc(signal_path: str, noise_path: str, train=True) -> list[tuple[str, str, float, int]]:
    """generates a signal to noise file association from folders
    Allows defining a standard data association for reproducibility. 
    Args:
        - signal_path: path to the signal folder
        - noise_path: path to the noise_folder
        - train: boolean (use training set)
    Returns:
        - a list of tuples, where each tuple contains:
          (signal file, noise file, random SNR, random event shift)

    """

    if train:
        signal_files = glob.glob(f"{signal_path}/train/**/*.npz", recursive=True)
        noise_files = glob.glob(f"{noise_path}/train/**/*.npz", recursive=True)

    else:
        signal_files = glob.glob(f"{signal_path}/validation/**/*.npz", recursive=True)
        noise_files = glob.glob(f"{noise_path}/validation/**/*.npz", recursive=True)

    # shuffle
    random.shuffle(signal_files)
    random.shuffle(noise_files)

    assoc = []
    for i in range(len(signal_files)):
        n = np.random.randint(0, len(noise_files))
        snr_random = np.random.uniform(0.2,1.5)
        event_shift = np.random.randint(1000,6000)
        assoc.append((signal_files[i], noise_files[n], snr_random, event_shift))

    return assoc


class InputSignals(th.utils.data.Dataset):

    def __init__(self, signal_noise_association:list):
        """
        Args:
            signal_noise_association: a list containing tuples for signal and noise filenames
        """

        self.signal_noise_assoc = signal_noise_association
        self.signal_length = 6120

    def __len__(self) -> int:
        return len(self.signal_noise_assoc)

    def __getitem__(self, idx: int) -> th.Tensor:

        while True:
            
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

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                idx = (idx + 1) % len(self.signal_noise_assoc)
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

            signal_std = np.std(eq_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)  
            noise_std = np.std(noise_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)
            snr_original = signal_std / noise_std
            # change the SNR
            noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
            eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
            noisy_eq = eq_stacked + noise_stacked # recombine

            return noisy_eq
        

class EventMasks(th.utils.data.Dataset):

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

        while True:

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

            if (
                Z_noise.shape != N_noise.shape
                or Z_noise.shape != E_noise.shape
                or N_noise.shape != E_noise.shape
            ):
                idx=(idx + 1) % len(self.signal_noise_assoc)
                continue

            noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

            signal_std = np.std(eq_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)  
            noise_std = np.std(noise_stacked[:,6000-event_shift:6500-event_shift], axis=1).reshape(-1,1)
            snr_original = signal_std / noise_std
            # change the SNR
            noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
            eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR

            stft = keras.ops.stft(eq_stacked, self.frame_length, self.frame_step, self.fft_size)
            stft_eq = np.concatenate([stft[0],stft[1]], axis=0)

            stft = keras.ops.stft(noise_stacked, self.frame_length, self.frame_step, self.fft_size)
            stft_noise = np.concatenate([stft[0],stft[1]], axis=0)

            mask = np.abs(stft_eq) / (np.abs(stft_noise) + np.abs(stft_eq) + 1e-4)
            
            return mask
        

class CombinedDeepDenoiserDataset(th.utils.data.Dataset):
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

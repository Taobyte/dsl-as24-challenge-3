import torch as th
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_dataloaders(signal_path: str, noise_path: str, batch_size: int) -> tuple[DataLoader, DataLoader]:

    signal_train_path = signal_path + "/train"
    signal_validation_path = signal_path + "/validation"

    noise_train_path = noise_path + "/train"
    noise_validation_path = noise_path + "/validation"

    train_dataset = SeismicDataset(signal_train_path, noise_train_path, randomized=True)
    validation_dataset = SeismicDataset(signal_validation_path, noise_validation_path, randomized=True)
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_dl, validation_dl



class SeismicDataset(Dataset):

    def __init__(self, signal_folder_path: str, noise_folder_path: str, randomized = False):

        """
        Args:
            signal_folder_path (str): Path to earthquake signal folder containing .npz files.
            noise_folder_path (str): Path to noise folder containing .npz files.
        """

        self.signal_folder_path = signal_folder_path
        self.noise_folder_path = noise_folder_path
        self.eq_signal_files = glob.glob(f'{signal_folder_path}/**/*.npz', recursive=True)
        self.noise_files = glob.glob(f'{noise_folder_path}/**/*.npz', recursive=True)
        self.randomized = randomized


    def __len__(self) -> int:
        return len(self.eq_signal_files)

    def __getitem__(self, idx) -> tuple[th.Tensor, th.Tensor]:

        eq_path = self.eq_signal_files[idx]
        eq = np.load(eq_path, allow_pickle=True)
        eq_name = (os.path.splitext(os.path.basename(eq_path)))[0]

        noise_idx = np.random.randint(0, len(self.noise_files))
        noise_path = self.noise_files[noise_idx]
        noise = np.load(noise_path, allow_pickle=True)
        noise_name = (os.path.splitext(os.path.basename(noise_path)))[0]

        noise_seq_len = len(noise['noise_waveform_Z'])

        assert noise_seq_len >= 6000

        eq_start = 0
        noise_start = 0
        if self.randomized:
            eq_start = np.random.randint(low = 0, high = 6000)
            noise_start = np.random.randint(low = 0, high = max(noise_seq_len - 6000, 1))

        Z_eq = eq['earthquake_waveform_Z'][eq_start:eq_start+6000]
        N_eq = eq['earthquake_waveform_N'][eq_start:eq_start+6000]
        E_eq = eq['earthquake_waveform_E'][eq_start:eq_start+6000]
        event = np.stack([Z_eq, N_eq, E_eq], axis=0)
        eq_tensor = th.from_numpy(event)

        Z_noise = noise['noise_waveform_Z'][noise_start:noise_start+6000]
        N_noise = noise['noise_waveform_N'][noise_start:noise_start+6000]
        E_noise = noise['noise_waveform_E'][noise_start:noise_start+6000]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)
        noise_tensor = th.from_numpy(noise_stacked)


        # tensor_normalized = eq_tensor / eqtensor.abs().max()

        p_wave_start = 6000 - eq_start
        # , p_wave_start, eq_name, noise_name

        noisy_eq = eq_tensor + noise_tensor

        return noisy_eq / noisy_eq.abs().max(), eq_tensor / eq_tensor.abs().max()
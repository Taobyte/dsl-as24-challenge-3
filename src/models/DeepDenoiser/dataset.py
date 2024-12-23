import glob
import logging
import random

import omegaconf
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from numpy import ndarray

from src.utils import Mode, Model
from src.stft import get_mask
from torch import Tensor
from typing import Union

logger = logging.getLogger()


def get_dataloaders_pytorch(
    cfg: omegaconf.DictConfig,
    return_test=False,
    subset=None,
    model: Model = Model.DeepDenoiser,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    if return_test:
        test_dataset = EQDataset(cfg.user.data.filename, Mode.TEST)
        test_dl = th.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.plot.n_examples,
            shuffle=False,
            num_workers=1,
        )
        return test_dl

    if cfg.random:
        train_dataset = EQRandomDataset(
            cfg.user.data.signal_path,
            cfg.user.data.noise_path,
            cfg.trace_length,
            cfg.snr_lower,
            cfg.snr_upper,
            Mode.TRAIN,
            model=model,
        )
        val_dataset = EQRandomDataset(
            cfg.user.data.signal_path,
            cfg.user.data.noise_path,
            cfg.trace_length,
            cfg.snr_lower,
            cfg.snr_upper,
            Mode.VALIDATION,
            model=model,
        )

    else:
        assert model != Model.DeepDenoiser
        train_dataset = EQDataset(cfg.user.data.filename, Mode.TRAIN)
        val_dataset = EQDataset(cfg.user.data.filename, Mode.VALIDATION)

    if subset:
        train_dataset = Subset(train_dataset, indices=range(subset))
        val_dataset = Subset(train_dataset, indices=range(subset))

    train_dl = th.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size, num_workers=2
    )
    val_dl = th.utils.data.DataLoader(
        val_dataset, batch_size=cfg.model.batch_size, num_workers=2
    )

    return train_dl, val_dl


def get_signal_noise_assoc(
    signal_path: str,
    noise_path: str,
    mode: Mode,
    size_testset=1000,
    snr=lambda: np.random.uniform(0.1, 1.1),
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

        signal_files = glob.glob(f"{signal_path}/validation/**/*.npz", recursive=True) #[:-size_testset]
        noise_files = glob.glob(f"{noise_path}/validation/**/*.npz", recursive=True) #[:-size_testset]
    
    elif mode == Mode.TEST:

        signal_files = glob.glob(f"{signal_path}/**/*.npz", recursive=True)
        noise_files = glob.glob(f"{noise_path}/**/*.npz", recursive=True)
    
    else:
        assert False, f"Mode {mode} not supported!"

    # shuffle
    signal_files = signal_files + signal_files
    noise_files = noise_files + noise_files
    random.shuffle(signal_files)
    random.shuffle(noise_files)

    assoc = []
    for i in range(len(signal_files)):
        n = np.random.randint(0, len(noise_files))
        snr_random = snr()
        r = np.random.randint(0, 5)
        if r == 0 and not (mode == Mode.TEST):
            event_shift = np.random.randint(0,1000)
        else:
            event_shift = np.random.randint(2500,6000)
        assoc.append((signal_files[i], noise_files[n], snr_random, event_shift))

    return assoc


def load_traces_and_shift(
    assoc: list, trace_length: int
) -> tuple[ndarray, ndarray, ndarray]:
    eq_traces = []
    noise_traces = []
    shifts = []

    for eq_path, noise_path, _, eq_shift in assoc:
        eq = np.load(eq_path, allow_pickle=True)
        noise = np.load(noise_path, allow_pickle=True)

        Z_eq = eq["earthquake_waveform_Z"][eq_shift : eq_shift + trace_length]
        N_eq = eq["earthquake_waveform_N"][eq_shift : eq_shift + trace_length]
        E_eq = eq["earthquake_waveform_E"][eq_shift : eq_shift + trace_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["noise_waveform_Z"][:trace_length]
        N_noise = noise["noise_waveform_N"][:trace_length]
        E_noise = noise["noise_waveform_E"][:trace_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        eq_traces.append(eq_stacked)
        noise_traces.append(noise_stacked)
        shifts.append(eq_shift)

    eq_traces = np.array(eq_traces)
    noise_traces = np.array(noise_traces)
    shifts = np.array(shifts)

    return eq_traces, noise_traces, shifts


class EQDataset(Dataset):
    def __init__(self, filename, mode: Mode):
        self.mode = mode
        if mode == Mode.TRAIN:
            self.file_noise = np.load(filename + "train_eq_005.npy", allow_pickle=True)
            self.file_eq = np.load(filename + "train_noise_005.npy", allow_pickle=True)
        elif mode == Mode.VALIDATION:
            self.file_noise = np.load(filename + "val_eq_005.npy", allow_pickle=True)
            self.file_eq = np.load(filename + "val_noise_005.npy", allow_pickle=True)
        elif mode == Mode.TEST:
            self.file_noise = np.load(filename + "tst_noise_001.npy", allow_pickle=True)
            self.file_eq = np.load(filename + "tst_eq_001.npy", allow_pickle=True)
            self.assoc = np.load(filename + "tst_eq_assoc.npy", allow_pickle=True)
        else:
            raise NotImplementedError

        assert self.file_eq.shape == self.file_noise.shape

    def __len__(self) -> int:
        return len(self.file_noise)

    def __getitem__(self, index):
        if self.mode == Mode.TEST:
            return (self.file_eq[index], self.file_noise[index], self.assoc[index][3])
        return (self.file_eq[index], self.file_noise[index])


class EQRandomDataset(Dataset):
    def __init__(
        self,
        signal_path: str,
        noise_path: str,
        trace_length: int,
        snr_lower: float,
        snr_upper: float,
        mode: Mode,
        model: Model = Model.DeepDenoiser,
        n_fft: int = 127,
        hop_length: int = 16,
        win_length: int = 100,
        window: str = "hann_window",
    ):
        if mode == Mode.TRAIN:
            self.signal_files = glob.glob(
                f"{signal_path}/train/**/*.npz", recursive=True
            )
            self.noise_files = glob.glob(f"{noise_path}/train/**/*.npz", recursive=True)
        elif mode == Mode.VALIDATION:
            self.signal_files = glob.glob(
                f"{signal_path}/validation/**/*.npz", recursive=True
            )
            self.noise_files = glob.glob(
                f"{noise_path}/validation/**/*.npz", recursive=True
            )
        else:
            raise NotImplementedError

        self.trace_length = trace_length
        self.snr_lower = snr_lower
        self.snr_upper = snr_upper

        assert self.snr_lower <= self.snr_upper

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(th, window)(self.win_length)

        self.model = model

    def __len__(self) -> int:
        return len(self.signal_files)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        eq = np.load(self.signal_files[idx], allow_pickle=True)
        noise = np.load(self.noise_files[idx], allow_pickle=True)

        snr_random = np.random.uniform(self.snr_lower, self.snr_upper)
        eq_shift = np.random.randint(6000 - (self.trace_length - 500), 6000)

        Z_eq = eq["earthquake_waveform_Z"][eq_shift : eq_shift + self.trace_length]
        N_eq = eq["earthquake_waveform_N"][eq_shift : eq_shift + self.trace_length]
        E_eq = eq["earthquake_waveform_E"][eq_shift : eq_shift + self.trace_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        # noise_shift = np.random.randint(0, 17000 - self.trace_length) TODO: fix this

        Z_noise = noise["noise_waveform_Z"][: self.trace_length]
        N_noise = noise["noise_waveform_N"][: self.trace_length]
        E_noise = noise["noise_waveform_E"][: self.trace_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        max_val = max(np.max(np.abs(noise_stacked)), np.max(np.abs(eq_stacked))) + 1e-12
        eq_stacked /= max_val
        noise_stacked /= max_val

        signal_std = np.std(
            eq_stacked[:, 6000 - eq_shift : 6500 - eq_shift], axis=1
        ).reshape(-1, 1)
        noise_std = np.std(
            noise_stacked[:, 6000 - eq_shift : 6500 - eq_shift], axis=1
        ).reshape(-1, 1)
        snr_original = signal_std / (noise_std + 1e-12)

        noise_stacked = noise_stacked * snr_original
        eq_stacked = eq_stacked * snr_random

        eq_stacked = th.from_numpy(eq_stacked)
        noise_stacked = th.from_numpy(noise_stacked)

        if self.model == Model.DeepDenoiser:
            mask = get_mask(
                eq_stacked,
                noise_stacked,
            )
            return (eq_stacked, noise_stacked, mask)

        return eq_stacked, noise_stacked

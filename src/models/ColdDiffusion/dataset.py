import glob

import torch
import einops
import numpy as np
from numpy import ndarray
import torch

from utils import Mode
from models.DeepDenoiser.dataset import get_signal_noise_assoc


class CDiffDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.file = np.load(filename, allow_pickle=True)
        self.file = self.file[:80]
        print(f"Loaded dataset with shape {self.file.shape}")

    def __getitem__(self, index):
        return self.file[index]
    def __len__(self):
        return len(self.file)

class ColdDiffusionDataset(torch.utils.data.Dataset):

    def __init__(self, filename, shape):
        # shape args: TRAIN: (20230, 6, signal_length) VAL: (4681, 6, signal_length)
        self.memmap_file = np.memmap(filename, mode="r+", shape=shape, dtype='float32')
        self.memmap_file = self.memmap_file
        print("shaape", self.memmap_file.shape)

    def __getitem__(self, index):
        return (self.memmap_file[index,:3,:], self.memmap_file[index,3:,:]) 
    def __len__(self):
        return len(self.memmap_file)
    

class TestColdDiffusionDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        # self.memmap_file = np.load(filename, allow_pickle=True)
        self.assoc = np.load(filename + "tst_eq_001_assoc.npy", allow_pickle=True)
        self.file_noise = np.load(filename + "tst_noise_001.npy", allow_pickle=True)[:80]
        self.file_eq = np.load(filename + "tst_eq_001.npy", allow_pickle=True)[:80]
        assert self.file_eq.shape == self.file_noise.shape
        # print(f"Loaded dataset with shape {self.file_noise.shape}")


    def __getitem__(self, index):
        return (self.file_eq[index], self.file_noise[index], self.assoc[index][3]) 
    def __len__(self):
        return len(self.file_noise)  

def compute_train_dataset(signal_length, mode, memmap):

    signal_path = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/event"
    noise_path = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/noise"
    if mode == Mode.TRAIN:
        dataset_eq_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/train_eq_001.dat"
        dataset_noise_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/train_noise_001.dat"
    elif mode == Mode.TEST:
        dataset_eq_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/tst_eq_001.dat"
        dataset_noise_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/tst_noise_001.dat"
    else:
        dataset_eq_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/val_eq_001.dat"
        dataset_noise_name = "/home/tim/Documents/Data-Science_MSc/DSLab/earthquake_data/val_noise_001.dat"
    if not mode == Mode.TEST:
        assoc = get_signal_noise_assoc(signal_path, noise_path, mode, size_testset=1000,
                                   snr=lambda : np.random.uniform(0.2, 1.9))
    else: 
        assoc = get_signal_noise_assoc(signal_path, noise_path, mode, size_testset=1000,
                                    snr=lambda : 1.0)
    full = []
    earthquakes = []
    noises = []

    for (eq_file, noise_file, snr_random, event_shift) in assoc:
        eq = np.load(eq_file, allow_pickle=True)
        noise = np.load(noise_file, allow_pickle=True)
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + signal_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + signal_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=1)

        Z_noise = noise["noise_waveform_Z"][:signal_length]
        N_noise = noise["noise_waveform_N"][:signal_length]
        E_noise = noise["noise_waveform_E"][:signal_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=1)

        max_val = np.max(np.abs(eq_stacked)) + 1e-10
        eq_stacked = eq_stacked / max_val
        max_val = np.max(np.abs(noise_stacked)) + 1e-10
        noise_stacked = noise_stacked / max_val

        signal_std = np.std(eq_stacked[6000-event_shift:6500-event_shift, :], axis=0) 
        noise_std = np.std(noise_stacked[6000-event_shift:6500-event_shift,:], axis=0)        
        snr_original = signal_std / (noise_std + 1e-10)

        # change the SNR
        noise_stacked = noise_stacked * snr_original  # rescale noise so that SNR=1
        eq_stacked = eq_stacked * snr_random  # rescale event to desired SNR
        if np.isnan(noise_stacked).any():
            print("std", snr_original)
            print("shift", event_shift)
            print("len", len(Z_noise))
            print(Z_noise[6000-event_shift:6500-event_shift])


        output = np.stack([eq_stacked.T, noise_stacked.T], axis=0)
        output = einops.rearrange(output, "n d t -> (n d) t")
        
        earthquakes.append(eq_stacked)
        noises.append(noise_stacked)
        full.append(output)

    # return np.array(full)

    full = np.array(full)
    print(full.shape)
    earthquakes = einops.rearrange(np.array(earthquakes), "n t d -> n d t")
    noises = einops.rearrange(np.array(noises), "n t d -> n d t")
    print(f"Shapes of eq: {earthquakes.shape}, and noise: {noises.shape}")
    if memmap:
        pass
        # file = np.memmap(dataset_name, dtype=np.float32, mode="w+", shape=full.shape)
        # file[:] = full
        # file.flush
    else:
        np.save(dataset_eq_name, earthquakes, allow_pickle=True)
        np.save(dataset_noise_name, noises, allow_pickle=True)
        # np.save(dataset_name[:-4], full, allow_pickle=True)
    if mode == Mode.TEST:
        np.save(dataset_eq_name[:-4] + "_assoc", assoc, allow_pickle=True)

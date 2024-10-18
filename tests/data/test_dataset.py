from torch.utils.data import Dataset, DataLoader
from src.data import DeepDenoiserDataset, SeismicDataset

import numpy as np 
import torch as th

def test_dataset_length():
    signal_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    noise_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    dataset = SeismicDataset(signal_folder, noise_folder)
    assert len(dataset) == 20230 

def test_output_shape():
    signal_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    noise_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    dataset = SeismicDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    next_tensor, noise_tensor = next(iter(dataloader))
    print(next_tensor.shape)
    print(noise_tensor.shape)
    assert next_tensor.shape == (6,6000)


def test_dataset_length():
    signal_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    noise_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    assert len(dataset) == 20230 

def test_output_shape():
    signal_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    noise_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    next_tensor,_ = next(iter(dataloader))
    assert next_tensor.shape == (1,31,201,2)

def test_output_is_real_mask():
    signal_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    noise_folder = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    _, mask  = next(iter(dataloader))
    assert th.all(mask >= 0) and th.all(mask <= 1)






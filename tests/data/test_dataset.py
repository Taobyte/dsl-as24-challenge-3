from torch.utils.data import Dataset, DataLoader
from src.data import DeepDenoiserDataset, SeismicDataset, DeepDenoiserDatasetTest

import numpy as np
import torch as th
import scipy
import matplotlib.pyplot as plt


def test_dataset_length():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = SeismicDataset(signal_folder, noise_folder)
    assert len(dataset) == 20230


def test_output_shape():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = SeismicDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    next_tensor, noise_tensor = next(iter(dataloader))
    print(next_tensor.shape)
    print(noise_tensor.shape)
    assert next_tensor.shape == (6, 6000)


def test_dataset_length():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    assert len(dataset) == 20230


def test_output_shape():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    next_tensor, _ = next(iter(dataloader))
    assert next_tensor.shape == (1, 31, 201, 2)


def test_output_is_real_mask():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = DeepDenoiserDataset(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    _, mask = next(iter(dataloader))
    assert th.all(mask >= 0) and th.all(mask <= 1)


def test_visualize_deep_denoiser_dataloader():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )
    dataset = DeepDenoiserDatasetTest(signal_folder, noise_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    noisy_eq, ground_truth_mask, eq, noise = next(iter(dataloader))

    def inverse_stft(signal):
        t, time_domain_signal = scipy.signal.istft(
            signal,
            fs=100,
            nperseg=30,
            nfft=60,
            boundary="zeros",
        )

        return time_domain_signal

    fig, ax = plt.subplots(1, 4)
    ax[0].plot(range(3000), inverse_stft(noisy_eq[0, :, :, 0]))
    ax[1].plot(range(3000), eq.reshape((3000,)))
    ax[2].plot(range(3000), noise.reshape((3000,)))

    cax1 = ax[3].imshow(
        ground_truth_mask[0, :, :, 0], cmap="plasma", interpolation="none"
    )
    fig.colorbar(cax1, ax=ax[3])
    ax[3].invert_yaxis()

    plt.show()

    assert True

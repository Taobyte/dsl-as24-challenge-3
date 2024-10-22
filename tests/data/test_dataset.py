from torch.utils.data import Dataset, DataLoader
from src.data import DeepDenoiserDataset, SeismicDataset, DeepDenoiserDatasetTest, InputSignals, EventMasks, get_signal_noise_assoc

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

def test_input_assoc_list():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/"
    )

    assoc = get_signal_noise_assoc(signal_folder, noise_folder, train=False)

    assert len(assoc[0]) == 4 and isinstance(assoc[0][0], str) and isinstance(assoc[0][1], str) and isinstance(assoc[0][2], float) and isinstance(assoc[0][3], int)


def test_input_signals():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/"
    )

    assoc = get_signal_noise_assoc(signal_folder, noise_folder, train=False)

    input_signals = InputSignals(assoc)

    assert len(input_signals[0]) == 3

def test_output_event_masks():
    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/"
    )

    assoc = get_signal_noise_assoc(signal_folder, noise_folder, train=False)

    print(len(assoc))

    event_masks = EventMasks(assoc)

    assert event_masks[0].shape == (6,256,64)

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


def visualize_deep_denoiser_dataloader():
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

    fig, ax = plt.subplots(1, 5)
    ax[0].plot(range(3000), inverse_stft(noisy_eq[0, :, :, 0]))
    ax[1].plot(range(3000), eq.reshape((3000,)))
    ax[2].plot(range(3000), noise.reshape((3000,)))

    cax1 = ax[3].imshow(
        ground_truth_mask[0, :, :, 0], cmap="plasma", interpolation="none"
    )
    cax1 = ax[4].imshow(
        ground_truth_mask[0, :, :, 1], cmap="plasma", interpolation="none"
    )
    fig.colorbar(cax1, ax=ax[3])
    ax[3].invert_yaxis()
    fig.colorbar(cax1, ax=ax[4])
    ax[4].invert_yaxis()

    ax[0].set_title('Noisy earthquake signal')
    ax[1].set_title('Clean earthquake signal')
    ax[2].set_title('Noise')
    ax[3].set_title('Ground truth eq mask')
    ax[4].set_title('Ground truth noise mask')

    plt.show()

    assert True

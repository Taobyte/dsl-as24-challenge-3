import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

from src.models.WaveDecompNet.dataset import WaveDecompNetDataset
from torch.utils.data import DataLoader



def test_wave_decomp():

    np.random.seed(123)

    signal_path = '/cluster/scratch/ckeusch/data/signal/train'
    noise_path = '/cluster/scratch/ckeusch/data/noise/train'

    train_dataset = WaveDecompNetDataset(signal_path, noise_path, 2048)
    dl = DataLoader(train_dataset, batch_size=32)
    input, y_true = next(iter(dl))
    
    time = range(2048)
    _, axs = plt.subplots(3,3)

    for i in range(3):
        signal = y_true[0,0,:,i]
        noise = y_true[0,1,:,i]
        axs[0][i].plot(time, input[0,:,i])
        axs[1][i].plot(time, signal)
        axs[2][i].plot(time, noise)
    
    plt.savefig('./data.png')
    assert input.shape == (32, 2048, 3) and y_true.shape == (32,2,2048,3)





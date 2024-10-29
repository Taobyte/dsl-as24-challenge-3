
from src.data import WaveDecompDataset
from torch.utils.data import DataLoader


def test_wave_decomp():
    signal_path = '/cluster/scratch/ckeusch/dsl-as24-challenge-3/data/'
    signal_train_path = signal_path + "signal_train.pkl"
    noise_train_path = signal_path + "noise_train.pkl"
    train_dataset = WaveDecompDataset(signal_train_path, noise_train_path, 0.5, 2.0)
    dl = DataLoader(train_dataset, batch_size=32)
    assert next(iter(dl)).shape == (32, 2048, 3)
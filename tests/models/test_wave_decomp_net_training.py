import keras
import numpy as np
from torch.utils.data import DataLoader

from src.models.WaveDecompNet.wave_decomp_net import UNet1D, WaveDecompLoss
from src.data import TupleDataset

def test_unet1d_loss_dataloader():
    unet = UNet1D(n_layers=6)
    input_dim = (2048,3)
    dl = DataLoader(TupleDataset(input_dim), batch_size=32)
    unet.compile(loss=WaveDecompLoss(), optimizer=keras.optimizers.AdamW(learning_rate=0.001))
    print("compiled")
    loss = unet.fit(dl, epochs=1)
    assert True

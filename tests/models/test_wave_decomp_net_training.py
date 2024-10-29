from src.models.WaveDecompNet.wave_decomp_net import UNet1D, WaveDecompLoss
import numpy as np
from torch.utils.data import Dataset, DataLoader
import keras


def test_unet1d_loss():

    model = UNet1D(n_layers=6)
    model.compile(
        loss=WaveDecompLoss(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )

    gt_signal = np.zeros((32,2048, 3))
    gt_noise = np.zeros((32,2048, 3))
    gt = np.stack([gt_signal, gt_noise],axis=1)
    input = gt_signal + gt_noise
    
    model.fit(input, gt, epochs=5)
    assert model.evaluate(input, gt) >= 0.0

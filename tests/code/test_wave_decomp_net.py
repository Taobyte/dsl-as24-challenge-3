import keras
import numpy as np

from src.models.WaveDecompNet.wave_decomp_net import DownsamplingLayer, UpsamplingLayer, UNet1D, WaveDecompLoss

def test_downsampling_layer():
    down = DownsamplingLayer(8)
    input = np.zeros((32,3000, 3))
    output, _ = down(input)
    assert output.shape == (32,1500, 8)

def test_upsampling_layer():
    up = UpsamplingLayer(3)
    input = np.zeros((32,1500, 8))
    residuals = np.zeros((32,3000,8))
    output = up(input, residuals)
    assert output.shape == (32,3000, 3)

def test_unet1d():
    unet = UNet1D(n_layers=6)
    input = np.zeros((32,2048,3))
    output = unet(input)
    assert output[:,0,:,:].shape == (32,2048,3) and output[:,1,:,:].shape == (32,2048,3)

def test_unet1d_loss():
    unet = UNet1D(n_layers=6)
    gt_signal = np.zeros((32,2048, 3))
    gt_noise = np.zeros((32,2048, 3))
    gt = np.stack([gt_signal, gt_noise],axis=1)
    input = gt_signal + gt_noise
    output = unet(input)
    loss = WaveDecompLoss()
    loss_value = loss(gt, output)
    assert loss_value >= 0.0


def test_keras_lstm():
    input = np.zeros((32,64,256))
    lstm = keras.layers.LSTM(256, return_sequences=True)
    output = lstm(input)
    assert output.shape == (32,64,256)
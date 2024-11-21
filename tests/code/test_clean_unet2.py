import torch

from src.models.CleanUNet.stft_loss import stft

def test_stft():

    fft_size=126
    shift_size=24
    win_length=100
    window="hann_window"

    window_tensor = getattr(torch, window)(win_length)

    input = torch.randn((32*3,6120))
    print(input.shape)
    output = stft(input, fft_size, shift_size, win_length, window_tensor)

    assert output.shape == (96, 256, 64)


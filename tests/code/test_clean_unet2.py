import torch
import pytest
from hydra import initialize, compose

from src.models.CleanUNet.stft_loss import stft
from src.models.CleanUNet2.clean_specnet import CleanSpecNet
from src.models.CleanUNet2.clean_unet2_model import CleanUnet2


def test_stft():
    fft_size = 126
    shift_size = 24
    win_length = 100
    window = "hann_window"

    window_tensor = getattr(torch, window)(win_length)

    input = torch.randn((32 * 3, 6120))
    print(input.shape)
    output = stft(input, fft_size, shift_size, win_length, window_tensor)

    assert output.shape == (96, 256, 64)


def test_stft_dns_dataset():
    fft_size = 1024
    shift_size = 256
    win_length = 1024
    window = "hann_window"

    window_tensor = getattr(torch, window)(win_length)

    input = torch.randn((1, 160000))
    print(input.shape)
    output = stft(input, fft_size, shift_size, win_length, window_tensor)

    assert output.shape == (96, 256, 64)


def test_clean_spec_net():
    specnet = CleanSpecNet(channels_input=3, channels_output=3)

    input_shape = (1, 3, 6120)
    inputs = torch.randn(input_shape)

    output = specnet(inputs)

    assert output.shape == (1, 3, 64, 256)


def test_clean_unet2():
    with initialize(config_path="../../src/conf/"):
        cfg = compose(config_name="config")

    model = CleanUnet2(cfg)
    input = torch.randn((1, 3, 6120))
    output = model(input)

    assert True

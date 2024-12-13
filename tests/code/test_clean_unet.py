import torch

from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch


def test_lstm_bottleneck():
    model = CleanUNetPytorch(bottleneck="lstm")

    input = torch.randn((32, 3, 4096))

    output = model(input)

    assert output.shape == input.shape

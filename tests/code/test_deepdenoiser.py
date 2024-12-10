import torch

from src.models.DeepDenoiser.deep_denoiser_pytorch import (
    DownsamplingBlock,
    UpsamplingBlock,
    DeepDenoiser,
)


def test_downsampling():
    channel_base = 8
    n_layers = 5
    dims = [channel_base * 2**i for i in range(n_layers)]

    for dim in dims:
        block = DownsamplingBlock(dim, 0.0)
        input = torch.randn((8, dim, 16, 16))
        output, skip = block(input)
        assert output.shape == (8, int(2 * dim), 8, 8)


def test_upsampling():
    channel_base = 8
    n_layers = 5
    dims = [channel_base * 2**i for i in range(n_layers)]
    dims = dims[::-1]
    for dim in dims:
        block = UpsamplingBlock(dim, 0.0)
        input = torch.randn((8, int(2 * dim), 8, 8))
        skip = torch.randn((8, dim, 16, 16))
        output = block(input, skip)
        assert output.shape == (8, dim, 16, 16)


def test_deepdenoiser():
    model = DeepDenoiser(3, 5, 8, 0.0)
    input = torch.randn((1, 3, 6120))
    output = model(input)
    assert output.shape == (1, 6, 64, 256)

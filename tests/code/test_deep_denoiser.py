import numpy as np

from src.models.DeepDenoiser.deep_denoiser_model import UNET, DownsamplingLayer, UpsamplingLayer


def test_downsampling_before_max_pooling():
    layer = DownsamplingLayer(8)
    current_input = np.zeros((1,31,201,2))
    after_max, before_max = layer(current_input)
    assert before_max.shape == (1,31,201,8)

def test_downsampling_after_max_pooling():
    layer = DownsamplingLayer(8)
    current_input = np.zeros((1,31,201,2))
    after_max, before_max = layer(current_input)
    assert after_max.shape == (1,16,101,8)

def test_downsampling_after_max_pooling_snd():
    layer = DownsamplingLayer(16)
    current_input = np.zeros((1,16,101,8))
    after_max, before_max = layer(current_input)
    assert after_max.shape == (1, 8, 51,16)

def test_downsampling_after_max_pooling_third():
    layer = DownsamplingLayer(32)
    current_input = np.zeros((1,8, 51,16))
    after_max, before_max = layer(current_input)
    assert after_max.shape == (1, 4, 26,32)

def test_downsampling_after_max_pooling_forth():
    layer = DownsamplingLayer(64)
    current_input = np.zeros((1, 4, 26,32))
    after_max, before_max = layer(current_input)
    assert after_max.shape == (1, 2, 13, 64)

def test_downsampling_after_max_pooling_fifth():
    layer = DownsamplingLayer(128)
    current_input = np.zeros((1, 2, 13, 64))
    after_max, before_max = layer(current_input)
    assert after_max.shape == (1, 1, 7, 128)

def test_downsampling_after_max_pooling_sixth():
    layer = DownsamplingLayer(256, conv_pool=False)
    current_input = np.zeros((1, 1, 7, 128))
    output_downsampling = layer(current_input)
    assert output_downsampling.shape == (1, 1, 7, 256)


def test_upsampling_first():

    first_layer = DownsamplingLayer(128)
    current_input = np.zeros((1, 2, 13, 64))
    after_max, before_max = first_layer(current_input) 

    layer = DownsamplingLayer(256, conv_pool=False)
    output_downsampling = layer(after_max)

    upsampling_layer = UpsamplingLayer(128, change_concat_dim=True)
    output = upsampling_layer(output_downsampling,before_max)

    assert output.shape == (1,2,13,128)

def test_upsampling_second():

    d_current = DownsamplingLayer(64)
    current_input = np.zeros((1, 4, 26,32))
    after_max, before_max = d_current(current_input)

    input = np.zeros((1,2,13,128))
    upsampling_layer = UpsamplingLayer(64, change_concat_dim=False)
    output = upsampling_layer(input,before_max)

    assert output.shape == (1, 4, 26, 64)


def test_upsampling_third():

    d_current = DownsamplingLayer(32)
    current_input = np.zeros((1,8, 51,16))
    after_max, before_max = d_current(current_input)

    input = np.zeros((1, 4, 26, 64))
    upsampling_layer = UpsamplingLayer(32, change_concat_dim=True)
    output = upsampling_layer(input,before_max)

    assert output.shape == (1, 8, 51, 32)

def test_upsampling_forth():

    d_current = DownsamplingLayer(16)
    current_input = np.zeros((1,16,101,8))
    after_max, before_max = d_current(current_input)

    input = np.zeros((1, 8, 51, 32))
    upsampling_layer = UpsamplingLayer(16, change_concat_dim=True)
    output = upsampling_layer(input,before_max)

    assert output.shape == (1, 16, 101, 16)

def test_upsampling_forth():

    d_current = DownsamplingLayer(8)
    current_input = np.zeros((1,31,201,2))
    after_max, before_max = d_current(current_input)

    input = np.zeros((1, 16, 101, 16))
    upsampling_layer = UpsamplingLayer(8, change_concat_dim_both=True)
    output = upsampling_layer(input,before_max)

    assert output.shape == (1, 31, 201, 8)


def test_unet():
    u_net = UNET()
    input = np.zeros((1,31,201,2))
    output = u_net(input)

    assert output.shape == input.shape


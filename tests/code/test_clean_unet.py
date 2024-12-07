import numpy as np
import keras

from src.models.CleanUNet.utils import FeedForward, PositionalEncoding, EncoderLayer, TransformerEncoder, CleanUNetLoss, ChannelAttentionBlock, TemporalAttentionBlock, RAGLUDown, RAGLUUp, CleanUNetInitializer
from src.models.CleanUNet.clean_unet_model import CleanUNet
from src.models.CleanUNet.clean_unet2_model import baseline_unet

def test_feed_forward():

    input = np.zeros((32,64,256))
    mlp = FeedForward(256, 128, dropout=0.1)
    output = mlp(input)
    assert output.shape == (32, 64, 256)

def test_pos_encoding():

    input = np.zeros((32,64,256))
    pos_encoding = PositionalEncoding(256, 64)
    output = pos_encoding(input)
    assert output.shape == (32, 64, 256)

def test_encoder_layer():
    
    input = np.zeros((32,64,512))
    enc_layer = EncoderLayer(512, 2048, 8, 64, 64, dropout=0.1)
    output, attn = enc_layer(input)
    assert output.shape == (32, 64, 512) and attn.shape == (32, 8, 64, 64)

def test_transformer_encoder():

    input = np.zeros((32,64,512))
    transformer = TransformerEncoder()
    output = transformer(input)
    assert output.shape == (32, 64, 512)

def test_clean_unet_lstm():

    input = np.zeros((32,2048,3))
    unet = CleanUNet(3,3, bottleneck="lstm")
    output = unet(input)
    assert output.shape == (32,2048,3)

def test_clean_unet_loss():
    
    y_pred = np.zeros((32,6120, 3))
    y_true = np.zeros((32,6120, 3))

    loss = CleanUNetLoss()

    output = loss(y_true, y_pred)

    print(output.numpy())

    assert output >= 0.0

def test_channel_attention():
    n_channels = 64
    input_shape = (32, 128, n_channels)
    input = np.zeros(input_shape)
    layer = ChannelAttentionBlock(n_channels)
    output = layer(input)

    assert output.shape == input_shape

def test_temporal_attention():
    n_channels = 64
    input_shape = (32, 128, n_channels)
    input = np.zeros(input_shape)
    layer = TemporalAttentionBlock(3)
    output = layer(input)

    assert output.shape == input_shape

def test_raglu_down():

    n_channels = 64
    T = 128
    input_shape = (32, T, n_channels)
    input = np.zeros(input_shape)

    layer = RAGLUDown(n_channels, 3, 2, CleanUNetInitializer(123))
    output = layer(input)

    assert output.shape == (32, T // 2, n_channels)

def test_raglu_up():

    n_channels = 64
    T = 128
    input_shape = (32, T, n_channels)
    input = np.zeros(input_shape)

    layer = RAGLUUp(n_channels, n_channels//2, 3, 2, CleanUNetInitializer(123))
    output = layer(input)

    assert output.shape == (32, int(T * 2), n_channels // 2)


def test_baseline_unet():

    model = baseline_unet(6120,[1,2,4,8], 8, 3)

    input = np.zeros((1, 6120, 3))
    output = model(input)
    assert output.shape == input.shape

def test_randomization_tuner():

    input = np.zeros((1,3,6120))
    choices = [8, 16, 32, 64, 128, 256]

    for a in choices:
        for b in choices:
            for c in [fft_length for fft_length in choices if fft_length >= a]:
                _, _ = keras.ops.stft(input, a, b, c)
    
    assert True



import numpy as np

from src.models.CleanUNet.clean_unet_model import FeedForward, PositionalEncoding, EncoderLayer, TransformerEncoder, CleanUNet, CleanUNetLoss

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

def test_clean_unet():

    input = np.zeros((32,6120,3))
    unet = CleanUNet(3,3)
    output = unet(input)
    assert output.shape == (32,6120,3)

def test_clean_unet_loss():
    
    y_pred = np.zeros((32,6120, 3))
    y_true = np.zeros((32,6120, 3))

    loss = CleanUNetLoss()

    output = loss(y_true, y_pred)

    print(output.numpy())

    assert output >= 0.0


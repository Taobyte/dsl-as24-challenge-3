import keras
from src.models.CleanUNet.utils import TransformerEncoderChollet, PositionalEncoding

def baseline_model(seq_length: int, kernel_size: int = 3):

    inputs = keras.Input(shape=(seq_length, 3))
    x = keras.layers.Conv1D(1, kernel_size=kernel_size, padding="same")(inputs) # (B, 6120, 1)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(seq_length, activation="relu")(x) 
    outputs = keras.layers.Conv1D(3, kernel_size=3, padding="same")(x)

    model = keras.Model(inputs, outputs)

    return model

def baseline_unet(seq_length: int, channel_dims: list[int], channel_base: int, kernel_size: int = 3):

    dense_dim = 128
    n_heads = 4

    skip_connections = []
        
    inputs = keras.Input(shape=(seq_length, 3))

    # inital conv
    x = keras.layers.Conv1D(channel_base, kernel_size=7, activation=None, padding="same")(inputs)

    residual = x 
    # encoder 
    for i in range(len(channel_dims)):
        x = keras.layers.Conv1D(channel_dims[i] * channel_base, kernel_size, strides=1, activation=None, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.sigmoid(x) * x
        x = keras.layers.Conv1D(channel_dims[i] * channel_base, kernel_size, strides=1, activation=None, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.sigmoid(x) * x
        x = x + keras.layers.Conv1D(channel_dims[i] * channel_base, kernel_size, strides=1, activation=None, padding="same")(residual)
        x = keras.layers.MaxPool1D(3, 2, padding="same")(x)
        residual = x
        skip_connections.append(x)
    # bottleneck
    skip_connections = list(reversed(skip_connections))
    channel_dims = list(reversed(channel_dims))
    # x = keras.layers.Conv1D(channel_dims[-1] * channel_base, kernel_size, strides=1)(x)
    # x = keras.layers.Bidirectional(keras.layers.LSTM(channel_dims[0] * channel_base, return_sequences = True))(x)
    # encoding = PositionalEmbedding(seq_length // (2 ** len(channel_dims)), 10000, embed_dim)(x)
    encoding = PositionalEncoding(channel_dims[0] * channel_base, x.shape[1])(x)
    x = TransformerEncoderChollet(channel_dims[0] * channel_base, dense_dim, n_heads)(x + encoding)
    x = keras.layers.BatchNormalization()(x)

    # decoder
    for i in range(len(channel_dims)):
        T = min(skip_connections[i].shape[1], x.shape[1])
        x = x[:, :T,:] + skip_connections[i][:,:T,:]
        x = keras.layers.UpSampling1D(2)(x)
        residual = x
        x = keras.layers.Conv1DTranspose(channel_dims[i] * channel_base // 2, kernel_size=kernel_size, activation=None, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.sigmoid(x) * x
        x = keras.layers.Conv1DTranspose(channel_dims[i] * channel_base // 2, kernel_size=kernel_size, activation=None, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.sigmoid(x) * x
        x = x + keras.layers.Conv1D(channel_dims[i] * channel_base // 2, kernel_size, strides=1, activation=None, padding="same")(residual)

    outputs = keras.layers.Conv1D(3, kernel_size, strides=1, padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

            

class CleanUnet2(keras.Model):

    def __init__(self):
        super().__init__()

    
    def call(self, x):
        pass

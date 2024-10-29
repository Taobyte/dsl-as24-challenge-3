import keras 
from numpy import ndarray
import tensorflow as tf


@keras.saving.register_keras_serializable()
class WaveDecompLoss(keras.losses.Loss):
    def __init__(self, name= "wave_decomp_loss" , reduction='sum_over_batch_size'):
        super().__init__(name = name, reduction=reduction)

        self.mse = keras.losses.MeanSquaredError()

    def call(self, y_true: tuple[ndarray, ndarray], y_pred:tuple[ndarray, ndarray]) -> float:

        print(y_true.shape)
        print(y_pred.shape)
        
        return 0.5 * self.mse(y_true[:,0,:,:], y_pred[:,0,:,:]) + 0.5 * self.mse(y_true[:,1,:,:], y_pred[:,1,:,:])

    def get_config(self):
        config = super().get_config()
        return config



@keras.saving.register_keras_serializable()
class DownsamplingLayer(keras.layers.Layer):

    def __init__(self, channel_size:int, dropout:float=0.0, kernel_size=3):
        super().__init__(name=f"DownsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout = dropout

        self.conv1 = keras.layers.Conv1D(
            channel_size,
            kernel_size=kernel_size,
            strides=1,
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_last"
        )
        self.conv2 = keras.layers.Conv1D(
            channel_size,
            kernel_size=kernel_size,
            strides=1,
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_last"
        )

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.relu = keras.layers.ReLU()
        self.max_pool = keras.layers.MaxPool1D(3, strides=2, padding="same", data_format="channels_last")

    def call(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        down = self.max_pool(x)

        return down, x

@keras.saving.register_keras_serializable()
class UpsamplingLayer(keras.layers.Layer):

    def __init__(self, channel_size:int, dropout:float=0.0, kernel_size=3):
        super().__init__(name=f"UpsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout = dropout

        self.Tconv1 = keras.layers.Conv1DTranspose(
            channel_size,
            kernel_size=kernel_size,
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_last"
        )
        self.Tconv2 = keras.layers.Conv1DTranspose(
            channel_size,
            kernel_size=kernel_size,
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_last"
        )

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.relu = keras.layers.ReLU()
        self.upsampling = keras.layers.UpSampling1D(2)

    def call(self, x, residuals):

        x= self.upsampling(x)

        x = keras.layers.concatenate([x, residuals],axis=-1)

        x = self.Tconv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.Tconv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return x

@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):
    
    def __init__(self, channel_base:int, n_layers:int, dropout:float=0.0):
        super().__init__(name=f"Decoder")

        self.upsamplers = [
            UpsamplingLayer(channel_base*(2**(n_layers-1-i)), dropout, kernel_size=2 * (i + 1) + 1) for i in range(n_layers)
            ]

        self.output_layer = keras.layers.Conv1D(
            3, kernel_size=1, activation=None, padding="same", use_bias=True, data_format="channels_last"
        )

        self.residuals = []
    
    def call(self, x):
        
        for i, up in enumerate(self.upsamplers):
            x = up(x, self.residuals[-(i+1)])

        x = self.output_layer(x)

        return x


@keras.saving.register_keras_serializable()
class UNet1D(keras.models.Model):

    def __init__(self, n_layers:int=3, dropout:float=0.0, channel_base:int=8, bottleneck:str=None, **kwargs):
        super().__init__()

        self.batchnorm1 = keras.layers.BatchNormalization()

        self.downsamplers = [
            DownsamplingLayer(channel_base*(2**i), dropout, kernel_size=2 * (n_layers - i) + 1) for i in range(n_layers)
            ]
        
        self.bottleneck = keras.layers.LSTM(channel_base*(2**n_layers), enable_sequence=True)

        self.batchnorm_middle = keras.layers.BatchNormalization()
        self.dropout_layer = keras.layers.Dropout
        self.relu = keras.layers.ReLU()

        self.signal_decoder = Decoder(channel_base, n_layers)
        self.noise_decoder = Decoder(channel_base, n_layers)


    def call(self, x):

        x = self.batchnorm1(x)

        residuals = []
        for down in self.downsamplers:
            x, res = down(x)
            residuals.append(res)
        
        self.signal_decoder.residuals = residuals
        self.noise_decoder.residuals = residuals
        
        x = self.bottleneck(x)
        x = self.batchnorm_middle(x)
        x = self.relu(x)

        signal = self.signal_decoder(x)
        noise = self.noise_decoder(x)

        output = keras.ops.stack([signal, noise], axis=1)

        return signal, noise
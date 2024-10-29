import os
import glob
import random

import keras
import torch as th
import numpy as np

# model reimplementation
@keras.saving.register_keras_serializable()
class DownsamplingLayer(keras.layers.Layer):

    def __init__(self, channel_size:int, dropout:float=0.0):
        super().__init__(name=f"DownsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout = dropout

        self.conv1 = keras.layers.Conv2D(
            channel_size,
            kernel_size=(5,5),
            strides=(1,1),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.conv2 = keras.layers.Conv2D(
            channel_size,
            kernel_size=(3,3),
            strides=(1,1),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.relu = keras.layers.ReLU()
        self.max_pool = keras.layers.MaxPool2D(3, strides=(2,2), padding="same", data_format="channels_first")

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
    
    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channel_size": self.channel_size,
                "dropout": self.dropout,
            }
        )
        return config
    

@keras.saving.register_keras_serializable()
class UpsamplingLayer(keras.layers.Layer):

    def __init__(self, channel_size:int, dropout:float=0.0):
        super().__init__(name=f"UpsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout = dropout

        self.Tconv1 = keras.layers.Conv2DTranspose(
            channel_size,
            kernel_size=(5,5),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.Tconv2 = keras.layers.Conv2DTranspose(
            channel_size,
            kernel_size=(3,3),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.relu = keras.layers.ReLU()
        self.upsampling = keras.layers.UpSampling2D((2,2), data_format="channels_first")

    def call(self, x, residuals):

        x= self.upsampling(x)

        x = keras.layers.concatenate([x, residuals],axis=1)

        x = self.Tconv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.Tconv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channel_size": self.channel_size,
                "dropout": self.dropout,
            }
        )
        return config
    

@keras.saving.register_keras_serializable()
class UNet(keras.models.Model):

    def __init__(self, n_layers:int=3, dropout:float=0.0, channel_base:int=8, frame_length = 100, frame_step = 24, fft_size = 126, **kwargs):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.channel_base = channel_base
        self.frame_length = frame_length
        self.frame_step = frame_step 
        self.fft_size = fft_size
        
        self.downsamplers = [
            DownsamplingLayer(channel_base*(2**i), self.dropout) for i in range(self.n_layers)
            ]

        self.upsamplers = [
            UpsamplingLayer(self.channel_base*(2**(self.n_layers-1-i)), self.dropout) for i in range(self.n_layers)
            ]
        self.middle_conv = keras.layers.Conv2D(
            self.channel_base * (2**self.n_layers),
            kernel_size=(3,3),
            strides=(1,1),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.batchnorm_middle = keras.layers.BatchNormalization()
        self.dropout_layer = keras.layers.Dropout
        self.relu = keras.layers.ReLU()

        self.output_layer = keras.layers.Conv2D(
            6, kernel_size=1, activation="sigmoid", padding="same", use_bias=True, data_format="channels_first"
        )


    def call(self, x):

        # first the STFT
        x = keras.ops.stft(x, self.frame_length, self.frame_step, self.fft_size)
        x = keras.layers.concatenate([x[0],x[1]], axis=1)
        x = self.batchnorm1(x)

        residuals = []
        for down in self.downsamplers:
            x, res = down(x)
            residuals.append(res)
        x = self.middle_conv(x)
        x = self.batchnorm_middle(x)
        x = self.relu(x)

        for i, up in enumerate(self.upsamplers):
            x = up(x, residuals[-(i+1)])

        x = self.output_layer(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dropout": self.dropout,
                "channel_base": self.channel_base,
                "n_layers": self.n_layers,
                "frame_length": self.frame_length, 
                "frame_step": self.frame_step,
                "fft_size": self.fft_size
            }
        )
        return config
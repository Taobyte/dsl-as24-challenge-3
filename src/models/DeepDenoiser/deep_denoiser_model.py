import numpy as np
import os
import keras
from numpy import ndarray
import tensorflow as tf

@keras.saving.register_keras_serializable()
class CustomSoftmaxCrossEntropy(keras.losses.Loss):
    def __init__(self, name= "custom_softmax_cross_entropy" , reduction='sum_over_batch_size'):
        super().__init__(name = name, reduction=reduction)

    def call(self, y_true, y_pred):
        flat_logits = tf.reshape(y_pred, [-1, 2], name="logits")
        flat_labels = tf.reshape(y_true, [-1, 2], name="labels")
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_logits))

    def get_config(self):
        config = super().get_config()
        return config


@keras.saving.register_keras_serializable()
class DownsamplingLayer(keras.layers.Layer):

    def __init__(self, channel_size: int, dropout_rate=0.0, conv_pool=True):
        super().__init__(name=f"DownsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.rate = dropout_rate
        self.conv_pool = conv_pool

        self.conv = keras.layers.Conv2D(
            channel_size,
            kernel_size=(3, 3),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.batch_normalization_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()
        self.dropout_1 = keras.layers.Dropout(rate=self.rate)

        if self.conv_pool:
            self.pool = keras.layers.Conv2D(
                channel_size,
                kernel_size=(3, 3),
                strides=(2, 2),
                activation=None,
                padding="same",
                use_bias=False,
                data_format="channels_first"
            )
            self.batch_normalization_2 = keras.layers.BatchNormalization()
            self.relu_2 = keras.layers.ReLU()
            self.dropout_2 = keras.layers.Dropout(rate=self.rate)

    def call(self, x: ndarray) -> ndarray:

        x = self.conv(x)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)
        conv_output = self.dropout_1(x)

        if self.conv_pool:
            y = self.pool(conv_output)
            y = self.batch_normalization_2(y)
            y = self.relu_2(y)
            z = self.dropout_2(y)

            return z, conv_output

        return conv_output

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channel_size": self.channel_size,
                "dropout_rate": self.dropout_rate,
                "conv_pool": self.conv_pool,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class UpsamplingLayer(keras.layers.Layer):

    def __init__(
        self,
        channel_size: int,
        dropout_rate=0.0,
    ):
        super().__init__(name=f"UpsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout_rate = dropout_rate

        self.transposed_conv = keras.layers.Conv2DTranspose(
            channel_size,
            kernel_size=3,
            strides=(2, 2),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.batch_normalization_0 = keras.layers.BatchNormalization()
        self.relu_0 = keras.layers.ReLU()
        self.drop_out_0 = keras.layers.Dropout(rate=self.dropout_rate)

        self.concatenate_layer = keras.layers.Concatenate(axis=1)

        self.convolution_layer = keras.layers.Conv2D(
            channel_size,
            kernel_size=(3, 3),
            activation=None,
            padding="same",
            use_bias=False,
            data_format="channels_first"
        )
        self.batch_normalization_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()
        self.drop_out_1 = keras.layers.Dropout(rate=self.dropout_rate)

    def build(self, input_shape):
        batch_size = input_shape[0]

    def call(self, x: ndarray, skip_tensor: ndarray) -> ndarray:

        x = self.transposed_conv(x)
        x = self.batch_normalization_0(x)
        x = self.relu_0(x)
        x = self.drop_out_0(x)
        
        x = self.concatenate_layer([x, skip_tensor])

        x = self.convolution_layer(x)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)
        x = self.drop_out_1(x)

        return x

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channel_size": self.channel_size,
                "dropout_rate": self.dropout_rate
            }
        )
        return config


@keras.saving.register_keras_serializable()
class Unet2D(keras.Model):

    def __init__(self, dropout_rate=0.0, channel_base=8, n_layers=5, frame_length = 100, frame_step = 24, fft_size = 126, **kwargs):
        super().__init__(**kwargs)

        self.dropout_rate = dropout_rate
        self.channel_base = channel_base
        self.n_layers = n_layers

        self.frame_length = frame_length
        self.frame_step = frame_step 
        self.fft_size = fft_size

        self.down_sampling = [
            DownsamplingLayer(channel_base * (2**i)) for i in range(self.n_layers)
        ]

        self.last_down_sampling_layer = DownsamplingLayer(256, conv_pool=False)

        self.up_sampling = [UpsamplingLayer(channel_base * (2**(self.n_layers - i - 1)), dropout_rate) for i in range(self.n_layers)]

        self.output_layer = keras.layers.Conv2D(
            6, kernel_size=1, activation="sigmoid", padding="same", use_bias=True, data_format="channels_first"
        )


    def call(self, x: ndarray) -> ndarray:

        x = keras.ops.stft(x, self.frame_length, self.frame_step, self.fft_size)
        x = tf.concat([x[0],x[1]], axis=1)

        skip_tensors = []
        for layer in self.down_sampling:
            x, y0 = layer(x)
            skip_tensors.append(y0)

        x = self.last_down_sampling_layer(x)

        for i,layer in enumerate(self.up_sampling):
            x = layer(x, skip_tensors[self.n_layers - i - 1])

        output = self.output_layer(x)
        
        return output

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "channel_base": self.channel_base,
                "n_layers": self.n_layers,
                "frame_length": self.frame_length, 
                "frame_step": self.frame_step,
                "fft_size": self.fft_size
            }
        )
        return config

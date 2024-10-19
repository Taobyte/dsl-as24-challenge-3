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
            )
            self.batch_normalization_2 = keras.layers.BatchNormalization()
            self.relu_2 = keras.layers.ReLU()
            self.dropout_2 = keras.layers.Dropout(rate=self.rate)

    def call(self, x):

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
        change_concat_dim=False,
        change_concat_dim_both=False,
    ):
        super().__init__(name=f"UpsamplingLayer_{channel_size}")

        self.channel_size = channel_size
        self.dropout_rate = dropout_rate
        self.change_concat_dim = change_concat_dim
        self.change_concat_dim_both = change_concat_dim_both

        self.transposed_conv = keras.layers.Conv2DTranspose(
            channel_size,
            kernel_size=3,
            strides=(2, 2),
            activation=None,
            padding="same",
            use_bias=False,
        )
        self.batch_normalization_0 = keras.layers.BatchNormalization()
        self.relu_0 = keras.layers.ReLU()
        self.drop_out_0 = keras.layers.Dropout(rate=self.dropout_rate)

        self.concatenate_layer = keras.layers.Concatenate(axis=-1)

        self.convolution_layer = keras.layers.Conv2D(
            channel_size,
            kernel_size=(3, 3),
            activation=None,
            padding="same",
            use_bias=False,
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

        if self.change_concat_dim:
            x = self.concatenate_layer([x[:, :, :-1, :], skip_tensor])
        elif self.change_concat_dim_both:
            x = self.concatenate_layer([x[:, :-1, :-1, :], skip_tensor])
        else:
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
                "dropout_rate": self.dropout_rate,
                "change_concat_dim": self.change_concat_dim,
                "change_concat_dim_both": self.change_concat_dim_both,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class Unet2D(keras.Model):

    def __init__(self, dropout_rate=0.0, channel_base=8, n_layers=5, **kwargs):
        super().__init__(**kwargs)

        self.dropout_rate = dropout_rate
        self.channel_base = channel_base
        self.n_layers = n_layers

        self.down_sampling = [
            DownsamplingLayer(channel_base * (2**i)) for i in range(self.n_layers)
        ]

        self.last_down_sampling_layer = DownsamplingLayer(256, conv_pool=False)

        self.ul0 = UpsamplingLayer(128, dropout_rate, change_concat_dim=True)
        self.ul1 = UpsamplingLayer(64, dropout_rate, change_concat_dim=False)
        self.ul2 = UpsamplingLayer(32, dropout_rate, change_concat_dim=True)
        self.ul3 = UpsamplingLayer(16, dropout_rate, change_concat_dim=True)
        self.ul4 = UpsamplingLayer(8, dropout_rate, change_concat_dim_both=True)

        self.output_layer = keras.layers.Conv2D(
            2, kernel_size=1, activation=None, padding="same", use_bias=True
        )

        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, x: ndarray) -> ndarray:

        skip_tensors = []
        for layer in self.down_sampling:
            x, y0 = layer(x)
            skip_tensors.append(y0)

        x = self.last_down_sampling_layer(x)

        x = self.ul0(x, skip_tensors[4])
        x = self.ul1(x, skip_tensors[3])
        x = self.ul2(x, skip_tensors[2])
        x = self.ul3(x, skip_tensors[1])
        x = self.ul4(x, skip_tensors[0])

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
            }
        )
        return config

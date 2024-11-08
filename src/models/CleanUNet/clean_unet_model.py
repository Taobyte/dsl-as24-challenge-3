import keras
import numpy as np
import einops

@keras.saving.register_keras_serializable()
class CleanUNetInitializer(keras.initializers.Initializer):

    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None):
      return keras.random.normal(
          shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):  # To support serialization
      return {'mean': self.mean, 'stddev': self.stddev}


@keras.saving.register_keras_serializable()
class CleanUNetLoss(keras.losses.Loss):
    def __init__(
        self,
        signal_length=2048,
        frame_lengths=[64, 128, 32],
        frame_steps=[16, 32, 8],
        fft_sizes=[128, 256, 64],
        name="custom_clean_unet_loss",
        reduction="sum_over_batch_size",
        **kwargs
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.signal_length = signal_length
        self.frame_lengths = frame_lengths
        self.frame_steps = frame_steps
        self.fft_sizes = fft_sizes

        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):

        # reshape for sftf
        y_true = einops.rearrange(y_true, "b t c -> b c t")
        y_pred = einops.rearrange(y_pred, "b t c -> b c t")

        stft_loss = 0

        def compute_stft_magnitude(y, frame_length, frame_step, fft_size):
            real, imag = keras.ops.stft(
                y, frame_length, frame_step, fft_size
            )
            return keras.ops.sqrt(
                keras.ops.clip(real**2 + imag**2, x_min=1e-7, x_max=float("inf"))
            )
        
        for frame_length, frame_step, fft_size in zip(self.frame_lengths, self.frame_steps, self.fft_sizes):
            
            y_true_stft = compute_stft_magnitude(y_true, frame_length, frame_step, fft_size)  # B C W H e.g (32, 3, 256, 64)
            y_true_stft = einops.rearrange(y_true_stft, "b c w h -> (b c) w h") # (96, 256, 64)
            y_pred_stft = compute_stft_magnitude(y_pred, frame_length, frame_step, fft_size)  # B C W H
            y_pred_stft = einops.rearrange(y_pred_stft, "b c w h -> (b c) w h") 

            frobenius_loss = keras.ops.mean(
                keras.ops.norm(y_true_stft - y_pred_stft, ord="fro", axis=(1, 2))
                / (keras.ops.norm(y_true_stft, ord="fro", axis=(1, 2)) + 1e-8)
            )

            log_loss = self.mae(keras.ops.log(y_true_stft) + 1e-8, keras.ops.log(y_pred_stft) + 1e-8)
            stft_loss += frobenius_loss + (1.0 / self.signal_length) * log_loss

        return 0.5 * stft_loss + self.mae(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "signal_length": self.signal_length,
                "frame_lengths": self.frame_lengths,
                "frame_steps": self.frame_steps,
                "fft_sizes": self.fft_sizes
            }
        )

        return config


class FeedForward(keras.layers.Layer):

    def __init__(self, dim_in: int, dim_hidden: int, dropout: float):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dropout = dropout

        self.dense1 = keras.layers.Dense(dim_hidden, activation="relu")
        self.dense2 = keras.layers.Dense(dim_in)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x):

        r = x
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + r

        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim_in": self.dim_in,
                "dim_hidden": self.dim_hidden,
                "dropout": self.dropout,
            }
        )
        return config


class PositionalEncoding(keras.layers.Layer):

    def __init__(self, d_hid: int, n_position=200):
        super().__init__()

        self.d_hid = d_hid
        self.n_position = n_position

        self.table = self._get_table(d_hid, n_position)

    def _get_table(self, d_hid: int, n_position: int):

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return np.expand_dims(sinusoid_table, axis=0)

    def call(self, x):
        _, T, _ = x.shape
        return x + self.table[:, :T, :]

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update({"d_hid": self.d_hid, "n_position": self.n_position})
        return config


class EncoderLayer(keras.layers.Layer):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.w_qs = keras.layers.Dense(n_head * d_k, use_bias=True)
        self.w_ks = keras.layers.Dense(n_head * d_k, use_bias=True)
        self.w_vs = keras.layers.Dense(n_head * d_v, use_bias=True)
        self.attention_layer = keras.layers.MultiHeadAttention(
            n_head, key_dim=d_k, value_dim=d_v, dropout=dropout
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = FeedForward(d_model, d_inner, dropout)

    def call(self, x):

        r = x
        q = self.w_qs(x)
        k = self.w_ks(x)
        v = self.w_vs(x)

        x, attn = self.attention_layer(q, v, k, return_attention_scores=True)
        x = self.layer_norm(x)
        x = x + r
        
        x = self.feed_forward(x)

        return x, attn

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "d_model": self.d_model,
                "d_inner": self.d_inner,
                "n_head": self.n_head,
                "d_k": self.d_k,
                "d_v": self.d_v,
                "dropout": self.dropout,
            }
        )
        return config


class TransformerEncoder(keras.layers.Layer):

    def __init__(
        self,
        d_word_vec=512,
        n_layers=2,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.0,
        n_position=624,
        scale_emb=False,
    ):

        super().__init__()

        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.n_position = n_position
        self.scale_emb = scale_emb

        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x

        self.dropout = keras.layers.Dropout(dropout)
        self.layer_stack = [
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def call(self, x, return_attns=False):

        self_attention_list = []

        if self.scale_emb:
            x *= self.d_model**0.5
        x = self.dropout(self.position_enc(x))
        x = self.layer_norm(x)

        for enc_layer in self.layer_stack:
            x, attn = enc_layer(x)
            self_attention_list += [attn] if return_attns else []

        if return_attns:
            return x, self_attention_list
        return x

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "d_word_vec": self.d_word_vec,
                "n_layers": self.n_layers,
                "n_head": self.n_head,
                "d_k": self.d_k,
                "d_v": self.d_v,
                "d_model": self.d_model,
                "d_inner": self.d_inner,
                "dropout": self.dropout,
                "n_position": self.n_position,
                "scale_emb": self.scale_emb,
            }
        )
        return config


class GLU(keras.layers.Layer):

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def call(self, x):
        a, b = keras.ops.split(x, 2, axis=self.axis)
        return a * keras.activations.sigmoid(b)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


class CleanUNet(keras.Model):
    """CleanUNet architecture."""

    def __init__(
        self,
        channels_input=3,
        channels_output=3,
        seq_length=6120,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=5,
        stride=2,
        tsfm_n_layers=3,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
        name=None,
        **kwargs
    ):
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """

        super(CleanUNet, self).__init__(name=name, **kwargs)

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.seq_length = seq_length

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        self.encoder = []
        self.decoder = []

        """
        def conv_output_shape(input_shape, stride):
            return int(keras.ops.floor((input_shape - 1) / stride + 1).numpy())

        self.conv_shapes = []
        current_shape = seq_length
        for i in range(encoder_n_layers):
            res = conv_output_shape(current_shape, stride)
            self.conv_shapes.append(res)
            current_shape = res
        self.conv_shapes = list(reversed(self.conv_shapes))
        print(self.conv_shapes)
        """

        for i in range(encoder_n_layers):
            self.encoder.append(
                keras.Sequential(
                    [
                        keras.layers.Conv1D(
                            channels_H,
                            kernel_size,
                            stride,
                            activation="relu",
                            padding="same",
                        ),
                        keras.layers.Conv1D(channels_H * 2, 1),
                        GLU(axis=2),
                    ]
                )
            )
            channels_input = channels_H

            if i == 0:
                # no relu at end
                self.decoder.append(
                    keras.Sequential(
                        [
                            keras.layers.Conv1D(channels_H * 2, 1),
                            GLU(axis=2),
                            keras.layers.Conv1DTranspose(
                                channels_output, kernel_size, stride, padding="same"
                            ),
                        ]
                    )
                )
            else:
                self.decoder.insert(
                    0,
                    keras.Sequential(
                        [
                            keras.layers.Conv1D(channels_H * 2, 1),
                            GLU(axis=2),
                            keras.layers.Conv1DTranspose(
                                channels_output,
                                kernel_size,
                                stride,
                                activation="relu",
                                padding="same",
                            ),
                        ]
                    ),
                )
            channels_output = channels_H

            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # Transformer Bottleneck

        self.tsfm_conv1 = keras.layers.Conv1D(tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(
            d_word_vec=tsfm_d_model,
            n_layers=tsfm_n_layers,
            n_head=tsfm_n_head,
            d_k=tsfm_d_model // tsfm_n_head,
            d_v=tsfm_d_model // tsfm_n_head,
            d_model=tsfm_d_model,
            d_inner=tsfm_d_inner,
            dropout=0.0,
            n_position=0,
            scale_emb=False,
        )
        self.tsfm_conv2 = keras.layers.Conv1D(channels_output, kernel_size=1)

    def call(self, x):
        _, T, _ = x.shape

        # encoder
        skip_connections = []
        for downsampling_layer in self.encoder:
            x = downsampling_layer(x)
            skip_connections.append(x)
        
        skip_connections = skip_connections[::-1]

        x = self.tsfm_conv1(x)
        x = self.tsfm_encoder(x)
        x = self.tsfm_conv2(x)

        # decoder
        for i, upsampling_layer in enumerate(self.decoder):
            skip = skip_connections[i]
            _, skip_T, _ = skip.shape
            x = x[:, :skip_T, :] + skip  # adding instead of concatenation
            x = upsampling_layer(x)

        x = x[:, :T, :]

        return x

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channels_input": self.channels_input,
                "channels_output": self.channels_output,
                "channels_H": self.channels_H,
                "max_H": self.max_H,
                "encoder_n_layers": self.encoder_n_layers,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "seq_length": self.seq_length,
                "tsfm_n_layers": self.tsfm_n_layers,
                "tsfm_n_head": self.tsfm_n_head,
                "tsfm_d_model": self.tsfm_d_model,
                "tsfm_d_inner": self.tsfm_d_inner,
            }
        )

        return config

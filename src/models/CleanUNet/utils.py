import keras
# import keras_hub
import numpy as np
import einops
import jax 
import jax.numpy as jnp

@keras.saving.register_keras_serializable()
class CleanUNetInitializer(keras.initializers.Initializer):

    def __init__(self, seed:int):
        super().__init__()
        self.seed = seed
        self.he_uniform = keras.initializers.HeUniform(seed=seed)

    def __call__(self, shape, dtype=None):

        weights = self.he_uniform(shape)
        alpha = keras.ops.std(weights)

        output = weights / keras.ops.sqrt(alpha)

        return output

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "seed": self.seed
            }
        )

        return config



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
        self.frame_lengths = list(frame_lengths)
        self.frame_steps = list(frame_steps)
        self.fft_sizes = list(fft_sizes)

        self.mae = keras.losses.MeanAbsoluteError()
        self.mse = keras.losses.MeanSquaredError()
        self.msle = keras.losses.MeanSquaredLogarithmicError()
        self.mape = keras.losses.MeanAbsolutePercentageError()

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
                keras.ops.clip(real**2 + imag**2, x_min=1e-7, x_max=1e9)
            )
        
        
        for frame_length, frame_step, fft_size in zip(self.frame_lengths, self.frame_steps, self.fft_sizes):
            
            
            y_true_stft = compute_stft_magnitude(y_true, frame_length, frame_step, fft_size)  # B C W H e.g (32, 3, 256, 64)
            y_true_stft = einops.rearrange(y_true_stft, "b c w h -> (b c) w h") # (96, 256, 64)
            y_pred_stft = compute_stft_magnitude(y_pred, frame_length, frame_step, fft_size)  # B C W H
            y_pred_stft = einops.rearrange(y_pred_stft, "b c w h -> (b c) w h") 
            """
            frobenius_loss = keras.ops.mean(
                keras.ops.norm(y_true_stft - y_pred_stft, ord="fro", axis=(1, 2))
                / (keras.ops.norm(y_true_stft, ord="fro", axis=(1, 2)) + 1e-8)
            )
            """
            frobenius_loss = (1.0 / 100) * self.mape(y_true_stft, y_pred_stft)
            
            log_loss = self.mae(keras.ops.log(y_true_stft), keras.ops.log(y_pred_stft))

            stft_loss += frobenius_loss + (1.0 / self.signal_length) * log_loss
            
            # stft_loss += self.mape(y_true_stft, y_pred_stft)
        
        return 0.5 * stft_loss

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

    def __init__(self, d_hid: int, n_position=200, name="PositionalEncoding", **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)

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
        

class EncoderLayerFromKeras(keras.layers.Layer):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayerFromKeras, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        initializer = CleanUNetInitializer(123)

        # self attention mechanism
        self.attention = keras.layers.MultiHeadAttention(
            n_head, key_dim=d_k, dropout=dropout
        )
        self.dropout1 = keras.layers.Dropout(dropout)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        # feed forward network
        self.conv1 = keras.layers.Conv1D(filters=d_inner, kernel_size=1, activation="relu", kernel_initializer=initializer)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.conv2 = keras.layers.Conv1D(filters=d_model,kernel_size=1, kernel_initializer=initializer)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):

        residual1 = x

        x, attn = self.attention(x,x, return_attention_scores=True)
        x = self.dropout1(x)
        x = self.layer_norm1(x)

        residual2 = x + residual1

        x = self.conv1(residual2)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)

        output = x + residual2

        return output, attn

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
        
        # self.position_enc = keras_hub.layers.SinePositionEncoding()

        self.dropout = keras.layers.Dropout(dropout)
        self.layer_stack = [
            EncoderLayerFromKeras(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
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
        a, b = keras.ops.split(x, 2, axis=self.axis) # (B, T, 2*C) ->  2 x (B,T,C)
        return a * keras.activations.sigmoid(b)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config

class GLUDown(keras.layers.Layer):
    
    def __init__(self, channels_H, kernel_size, stride, initializer, name="GLUDown"):
        super().__init__(name=name)
        self.layer = keras.Sequential(
                        [
                            keras.layers.Conv1D(
                                channels_H,
                                kernel_size,
                                stride,
                                activation="relu",
                                padding="same",
                                kernel_initializer=initializer
                            ),
                            keras.layers.Conv1D(channels_H * 2, 1, kernel_initializer=initializer),
                            GLU(axis=2),
                        ]
                )
    
    def call(self, x):
        return self.layer(x)

class GLUUp(keras.layers.Layer):
    
    def __init__(self,channels_H, channels_output, kernel_size, stride, initializer, use_relu=False, name="GLUUp"):
        super().__init__(name=name)

        self.layer = keras.Sequential(
                        [
                            keras.layers.Conv1D(channels_H * 2, 1, kernel_initializer=initializer),
                            GLU(axis=2),
                            keras.layers.Conv1DTranspose(
                                channels_output,
                                kernel_size,
                                stride,
                                activation="relu" if use_relu else None,
                                padding="same",
                                kernel_initializer=initializer
                            )
                        ]
                    )
    
    def call(self,x):
        x = self.layer(x)
        return x

    
class ChannelAttentionBlock(keras.layers.Layer):
    def __init__(self, n_channels: int, reduction_ratio:int=16):
        super().__init__()

        self.mlp = keras.Sequential([
            keras.layers.Dense(n_channels // reduction_ratio, activation="relu"),
            keras.layers.Dense(n_channels)
        ])

    def call(self, x):
        # input shape x.shape == (B, T, C)
        max_pool = keras.ops.max(x, axis=1, keepdims=True)
        avg_pool = keras.ops.mean(x, axis=1, keepdims=True)

        max_pool_mlp = self.mlp(max_pool)
        avg_pool_mlp = self.mlp(avg_pool)

        scale = keras.activations.sigmoid(max_pool_mlp + avg_pool_mlp)
        return x * scale



class TemporalAttentionBlock(keras.layers.Layer):
    def __init__(self, kernel_size: int):
        super().__init__()

        self.batch_norm = keras.layers.BatchNormalization()
        self.conv_1d = keras.layers.Conv1D(1, kernel_size, 1, padding="same")
    
    def call(self, x):
        # input shape x.shape == (B, T, C)
        max_pool = keras.ops.max(x, axis=2, keepdims=True) # (B, T, 1)
        avg_pool = keras.ops.mean(x, axis=2, keepdims=True) # (B, T, 1)

        concat = keras.ops.concatenate([max_pool, avg_pool], axis=2) # (B, T, 2)

        conv_1d_res = self.conv_1d(concat) # (B, T, 1)
        conv_1d_res = self.batch_norm(conv_1d_res)

        mask = keras.activations.sigmoid(conv_1d_res)
        return x * mask



class RAGLUDown(keras.layers.Layer):

    def __init__(self, channels_H, kernel_size, stride, initializer, name="RAGLUDown"):
        super().__init__(name=name)

        self.conv_1 = keras.layers.Conv1D(channels_H, kernel_size, stride, activation="relu", padding="same", kernel_initializer=initializer)
        self.conv_2 = keras.layers.Conv1D(channels_H * 2, 1, kernel_initializer=initializer)

        self.channel_attention = ChannelAttentionBlock(channels_H)
        self.temporal_attention = TemporalAttentionBlock(kernel_size)
    
    def call(self, x):

        x = self.conv_1(x)
        residual = x

        x = self.conv_2(x)
        a, b = keras.ops.split(x, 2, axis=2)

        a = self.channel_attention(a)
        a = self.temporal_attention(a)
        a = a + residual

        return a * keras.activations.sigmoid(b)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config

class RAGLUUp(keras.layers.Layer):

    def __init__(self,channels_H, channels_output, kernel_size, stride, initializer, use_relu=False, name="RAGLUUp"):
        super().__init__(name=name)

        self.conv= keras.layers.Conv1D(channels_H * 2, 1, kernel_initializer=initializer)
        self.conv_trans = keras.layers.Conv1DTranspose(
                                channels_output,
                                kernel_size,
                                stride,
                                activation="relu" if use_relu else None,
                                padding="same",
                                kernel_initializer=initializer
                            )
        
        self.channel_attention = ChannelAttentionBlock(channels_H)
        self.temporal_attention = TemporalAttentionBlock(kernel_size)

    def call(self, x):

        residual = x 
        x = self.conv(x)
        a, b = keras.ops.split(x, 2, axis=2)
        a = self.channel_attention(a)
        a = self.temporal_attention(a)
        a = a + residual

        output = a * keras.activations.sigmoid(b)
        return self.conv_trans(output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"), 
            keras.layers.Dense(embed_dim),]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs):
        attention_output = self.attention(
        inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
    def call(self, inputs):
        length = inputs.shape[1] # (B, T, C)
        positions = keras.ops.arange(length)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
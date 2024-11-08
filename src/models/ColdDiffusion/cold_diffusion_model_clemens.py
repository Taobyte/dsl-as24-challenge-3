import keras
import numpy as np
import einops


@keras.saving.register_keras_serializable()
class Block(keras.layers.Layer):

    def __init__(self, n_filters: int, groups: int = 8):
        super().__init__()

        self.n_filters = n_filters
        self.groups = groups

        self.conv_layer = keras.layers.Conv1D(
            filters=n_filters, kernel_size=3, padding="same"
        )
        self.group_norm_layer = keras.layers.GroupNormalization(groups)

    def call(self, x, scale_shift=None):
        x = self.conv_layer(x)
        x = self.group_norm_layer(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # swish activation function = x * sigmoid(x)
        return x * keras.activations.sigmoid(x)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update({"n_filters": self.n_filters, "groups": self.groups})
        return config


@keras.saving.register_keras_serializable()
class ResnetBlock(keras.layers.Layer):

    def __init__(self, dim_in: int, n_filters: int, time_emb_dim=None, groups: int = 8):
        super().__init__()

        self.dim_in = dim_in
        self.n_filters = n_filters
        self.time_emb_dim = time_emb_dim
        self.groups = groups

        self.linear_layer = keras.layers.Dense(n_filters * 2) if time_emb_dim else None
        self.block1 = Block(n_filters, groups=groups)
        self.block2 = Block(n_filters, groups=groups)
        self.res_conv = (
            keras.layers.Conv1D(n_filters, kernel_size=1)
            if dim_in != n_filters
            else keras.layers.Identity()
        )

    def call(self, x, time_emb=None):

        scale_shift = None
        if time_emb is not None and self.linear_layer is not None:
            time_emb = self.linear_layer(time_emb)
            time_emb = time_emb * keras.activations.sigmoid(
                time_emb
            )  # swish act function
            time_emb = einops.rearrange(time_emb, "b c -> b 1 c")
            scale_shift = keras.ops.split(time_emb, 2, axis=2)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim_in": self.dim_in,
                "n_filters": self.n_filters,
                "time_emb_dim": self.time_emb_dim,
                "groups": self.groups,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class LinearAttention(keras.layers.Layer):
    pass


@keras.saving.register_keras_serializable()
class Attention(keras.layers.Layer):

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        hidden_dim = heads * dim_head
        self.to_qkv = keras.layers.Conv1D(
            filters=hidden_dim * 3, kernel_size=1, use_bias=False
        )
        self.multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=dim_head
        )
        self.to_out = keras.layers.Conv1D(filters=dim, kernel_size=1)

    def call(self, x):
        # B, T, C = x.shape
        q, k, v = keras.ops.split(self.to_qkv(x), 3, axis=2)
        output = self.multi_head_attention(q, v, k)
        output = self.to_out(output)
        return output

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update({"dim": self.dim, "heads": self.heads, "dim_head": self.dim_head})
        return config


@keras.saving.register_keras_serializable()
class SinusoidalPosEmb(keras.layers.Layer):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def call(self, x):
        half_dim = self.dim // 2
        emb = keras.ops.log(10000) / (half_dim - 1)
        emb = keras.ops.exp(
            keras.ops.cast(keras.ops.arange(half_dim), "float32") * -emb
        )
        emb = x[:, None] * emb[None, :]
        emb = keras.ops.concatenate((keras.ops.sin(emb), keras.ops.cos(emb)), axis=-1)
        return emb

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class GELU(keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return keras.activations.gelu(x)

    def get_config(self):
        config = super().get_config()
        return config


@keras.saving.register_keras_serializable()
class RMSNorm(keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.l2_normalization = keras.layers.UnitNormalization(axis=2)

    def build(self):
        self.g = self.add_weight(
            shape=(1, 1, self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="RMSNorm",
        )

    def call(self, x):
        x = self.l2_normalization(x)
        return x * self.g * (self.dim**0.5)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class NormAttentionBlock(keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.rms_norm_layer = RMSNorm(dim)
        self.attention_layer = Attention(dim)

    def call(self, x):
        h = self.rms_norm_layer(x)
        h = self.attention_layer(h)
        return x + h

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config


def Upsample(dim):
    return keras.Sequential(
        [keras.layers.UpSampling1D(size=2), keras.layers.Conv1D(dim, 3, padding="same")]
    )


@keras.saving.register_keras_serializable()
class Unet1D(keras.Model):

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dim = dim
        self.init_dim = dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.resnet_block_groups = resnet_block_groups

        self.initial_conv = keras.layers.Conv1D(
            filters=init_dim, kernel_size=7, padding="same"
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embedding
        time_dim = dim * 4
        self.time_dim = time_dim
        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = keras.Sequential(
            [
                sinu_pos_emb,
                keras.layers.Dense(time_dim),
                GELU(),
                keras.layers.Dense(time_dim),
            ]
        )

        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        # downsampling layers
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == num_resolutions - 1

            self.downs.append(
                [
                    ResnetBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    ResnetBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    NormAttentionBlock(dim_in),
                    (
                        keras.layers.Conv1D(
                            dim_out, kernel_size=4, strides=2, padding="same"
                        )
                        if not is_last
                        else keras.layers.Conv1D(dim_out, 3, padding="same")
                    ),
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = NormAttentionBlock(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == num_resolutions - 1

            self.ups.append(
                [
                    ResnetBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    ResnetBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    NormAttentionBlock(dim_out),
                    (
                        Upsample(dim_in)
                        if not is_last
                        else keras.layers.Conv1D(dim_in, 3, padding="same")
                    ),
                ]
            )

        self.out_dim = out_dim if out_dim else channels
        self.final_res_block = ResnetBlock(2 * dim, dim, time_emb_dim=time_dim)
        self.final_conv_block = keras.layers.Conv1D(self.out_dim, 1)

    def build(self, input_shape):
        B, T, C = input_shape 
        self.B = B

    # time should be included as a parameter
    def call(self, x):

        x = self.initial_conv(x)
        r = keras.ops.copy(x)

        # hack to train model without diffusion
        # time = keras.ops.convert_to_tensor(np.zeros(B))
        # t = self.time_mlp(time)
        t = keras.ops.convert_to_tensor(np.zeros((self.B, self.time_dim)))

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = keras.ops.concatenate((x, h.pop()), axis=2)
            x = block1(x, t)

            x = keras.ops.concatenate((x, h.pop()), axis=2)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = keras.ops.concatenate((x, r), axis=2)
        x = self.final_res_block(x, t)
        x = self.final_conv_block(x)

        return x

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dim": self.dim,
                "init_dim": self.init_dim,
                "dim_mults": self.dim_mults,
                "resnet_block_groups": self.resnet_block_groups,
            }
        )
        return config

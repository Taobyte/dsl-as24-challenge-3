import os
from functools import partial

os.environ["KERAS_BACKEND"] = "jax"
import keras
import einops

@keras.saving.register_keras_serializable()
class SinusoidalEmbeddings(keras.layers.Layer):
    """turns a time integer into a sinusoidal embedding vector
    Args:
        - out_dim: the dimension of the embedding vector 
    """

    def __init__(self, out_dim:int):
        super().__init__(name=f"SinusoidalEmbeddings_dim{out_dim}")

        self.dim = out_dim

    def call(self, time):
        half_dim = self.dim // 2
        embedding = keras.ops.log(1000) / (half_dim-1)
        embedding = keras.ops.exp(keras.ops.arange(half_dim) * -embedding)
        embedding = time[:,None] * embedding[None,:] # multiplies each time onto same embedding vector
        embedding = keras.ops.concatenate([keras.ops.sin(embedding), keras.ops.sin(embedding)], axis=-1)
        return embedding
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
    
@keras.saving.register_keras_serializable()
class PreNorm(keras.layers.Layer):
    def __init__(self, function):
        super().__init__(name=f"PreNorm")
        self.fct = function
        self.norm = keras.layers.LayerNormalization(axis=1, rms_scaling=True)

    def call(self, x):
        x = self.norm(x)
        return self.fct(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"fct":self.fct})
        return config
    
@keras.saving.register_keras_serializable()
class Residual(keras.layers.Layer):
    def __init__(self, function):
        super().__init__(name=f"Residual")
        self.fct = function

    def call(self, x, *args, **kwargs):
        return self.fct(x, *args, **kwargs) + x
    
    def get_config(self):
        config = super().get_config()
        config.update({"fct":self.fct})
        return config

@keras.saving.register_keras_serializable()
class Upsample1D(keras.layers.Layer):
    def __init__(self, out_filters:int):
        super().__init__(name=f"Upsample1D_f{out_filters}")
        self.out_filters = out_filters
        self.up = keras.layers.UpSampling1D(size=2)
        self.conv = keras.layers.Conv1D(out_filters, kernel_size=4, strides=1, padding="same",
                                        data_format="channels_first", use_bias=False)
        
    def call(self, x):
        x = einops.rearrange(x, "b d t -> b t d") # traspose
        x = self.up(x)
        x = einops.rearrange(x, "b t d -> b d t") # transpose back
        x = self.conv(x)
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({"out_filters":self.out_filters})
        return config
    
@keras.saving.register_keras_serializable()
class Downsample1D(keras.layers.Layer):
    def __init__(self, out_filters:int):
        super().__init__(name=f"Downsample1D_f{out_filters}")
        self.out_filters = out_filters
        self.down = keras.layers.Conv1D(out_filters, kernel_size=4, strides=2, padding="same",
                                        data_format="channels_first", use_bias=False)
        
    def call(self, x):
        return self.down(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({"out_filters":self.out_filters})
        return config

@keras.saving.register_keras_serializable()
class ResNetBlock(keras.layers.Layer):
    """ResNet Block composed of two Conv1Ds and a residual connection
    Args:
        - in_filters: input filter dimension
        - out_filters: output filter dimension
        - time_emb_dim: time embedding dimension
        - n_groups: Nr. of normalization groups (1 for LayerNorm)
        - kernel_size: convolutional kernel size
    """
    def __init__(self, in_filters, out_filters, time_emb_dim=None, n_groups=1, kernel_size=3):
        super().__init__(name=f"ResNetBlock_{in_filters}_to_{out_filters}")

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.time_emb_dim = time_emb_dim
        self.kernel_size = kernel_size
        self.n_groups = n_groups

        self.mlp = keras.layers.Dense(out_filters*2, activation=keras.activations.silu) if time_emb_dim else None

        self.conv1 = keras.layers.Conv1D(self.out_filters, self.kernel_size, padding="same", data_format="channels_first", activation=None)
        self.conv2 = keras.layers.Conv1D(self.out_filters, self.kernel_size, padding="same", data_format="channels_first", activation=None)

        self.act1 = keras.activations.silu
        self.act2 = keras.activations.silu

        if self.n_groups > 1:
            self.norm1 = keras.layers.GroupNormalization(groups=self.n_groups)
            self.norm2 = keras.layers.GroupNormalization(groups=self.n_groups)
        else:
            self.norm1 = keras.layers.LayerNormalization()
            self.norm2 = keras.layers.LayerNormalization()

        # residual projection
        if self.in_filters != self.out_filters:
            self.residual_projection = keras.layers.Conv1D(self.out_filters, kernel_size=1, data_format="channels_first", padding="same")
        else:
            self.residual_projection = keras.layers.Identity()

    def call(self, x, time_embedding=None):

        assert time_embedding.shape[-1] == self.time_emb_dim, "Custom Flag: time embeddings has unexpected dimension"
        assert x.shape[1] == self.in_filters, "Custom Flag, got unexpected number of input filters"
        scale_shift = None
        if (time_embedding is not None) and (self.mlp is not None):
            time_embedding = self.mlp(time_embedding)
            time_embedding = einops.rearrange(time_embedding, "b 1 c -> b c 1")
            scale_shift = keras.ops.split(time_embedding, 2, axis=1) # returns tuple

        h = self.conv1(x)
        h = self.norm1(h)
        # potential time shifting
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (1+scale) + shift

        self.act1(h)

        self.conv2(h)
        self.norm2(h)
        self.act2(h)

        return h + self.residual_projection(x)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_filters": self.in_filters,
                "out_filters": self.out_filters,
                "time_emb_dim": self.time_emb_dim,
                "kernel_size": self.kernel_size,
                "n_groups": self.n_groups,
            }
        )
        return config
    

@keras.saving.register_keras_serializable()
class AttentionBlock(keras.layers.Layer):
    """Attention block, computes attention scores between timesteps
    Args:
        - linear: whether this is a linear attention block or not
        - filters: dimension of the input / output vectors in the timeseries
        - n_heads: how many attention heads to use
        - dim_heads: dimension of the vectors in each head
    """

    def __init__(self, linear:bool, filters:int, n_heads:int=4, dim_heads:int=32):
        super().__init__(name=f"AttentionBlock_Lin{linear}_f{filters}_n{n_heads}_d{dim_heads}")

        self.linear = linear
        self.filters = filters
        self.n_heads = n_heads
        self.dim_heads = dim_heads
        hidden_dim = self.dim_heads * self.n_heads

        self.to_qkv = keras.layers.Conv1D(3*hidden_dim, kernel_size=1, use_bias=False, data_format="channels_first", padding="same")
        self.out_proj = keras.layers.Conv1D(self.filters, kernel_size=1, use_bias=False, data_format="channels_first", padding="same")
        if self.linear:
            self.kernel = lambda x: keras.ops.elu(x) + 1

    def call(self, x):

        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(qkv, "b (c h n) t -> n b h c t", n=3, h=self.n_heads)

        if self.linear:
            # using linear attention kernel from OG paper
            q = self.kernel(q)
            k = self.kernel(k)

            # notice matmul in dimension d, not Nr. of timesteps n
            cntxt = einops.einsum(k, v, "b h d t, b h e t -> b h d e")
            norm = einops.rearrange(einops.einsum(q, k, "b h d t, b h d t -> b h d"), "b h d -> b h 1 d")
            val = einops.einsum(q, cntxt, "b h d t, b h d e -> b h t e")
            val = val / (self.dim_heads**0.5) 
            val = val / norm 

        else:
            attn = einops.einsum(q, k, "b h d t, b h d s -> b h t s")
            attn = attn / (self.dim_heads**0.5)
            # attn is computed from each timestep to each other timestep (shape time x time)
            attn = keras.ops.softmax(attn, axis=-1)
            val = einops.einsum(attn, v, "b h t s, b h d s -> b h t d")

        
        # heads and dimension gets recombined
        val = einops.rearrange(val, "b h t d -> b (h d) t")
        out = self.out_proj(val)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "linear": self.linear,
                "filters": self.filters,
                "n_heads": self.n_heads,
                "dim_heads": self.dim_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class DiffusionUnet1D(keras.models.Model):

    def __init__(
            self,  
            dim, 
            dim_multiples:tuple=(1,2,4,8),
            in_dim:int=3,
            out_dim:int=3,
            attn_dim_head:int=32,
            attn_heads:int=4,
            resnet_norm_groups:int=8,

            ):
        
        super().__init__()

        self.dim = dim
        self.dim_multiples = dim_multiples
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim_head = attn_dim_head
        self.attn_heads = attn_heads
        self.resnet_norm_groups = resnet_norm_groups
        self.time_dim = self.dim * 4
        # new function classes to shorten code
        block_class = partial(ResNetBlock, n_groups=self.resnet_norm_groups, time_emb_dim=self.time_dim)
        keras_conv1d = partial(keras.layers.Conv1D, data_format="channels_first", padding="same")
        # dimensions are expanded (e.g. dim=3 and dim_multiples=(1,2,3) --> dimensions=[1,6,9])
        dimensions = [*map(lambda m: self.dim*m, self.dim_multiples)]
        self.dim_tuples = list(zip(dimensions[:-1], dimensions[1:])) # tuples (in_dim,out_dim) (e.g. [(1,6),(6,9)])


        time_sinu_embeds = SinusoidalEmbeddings(self.dim)
        self.time_mlp = keras.Sequential([
            time_sinu_embeds,
            keras.layers.Dense(self.time_dim, activation=keras.activations.gelu),
            keras.layers.Dense(self.time_dim, activation=None)
        ])
        
        self.init_conv = keras_conv1d(self.dim, 7)
        self.downs = []
        self.ups = []


        for i, (in_dim, out_dim) in enumerate(self.dim_tuples):
            is_last = (i >= len(self.dim_tuples)-1)
            self.downs.append([
                block_class(in_dim, in_dim),
                block_class(in_dim, in_dim),
                Residual(PreNorm(AttentionBlock(True, in_dim, self.attn_heads, self.attn_dim_head))),
                Downsample1D(out_dim) if not is_last else keras_conv1d(out_dim, 3)
            ])

        mid_dim = dimensions[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim)
        self.mid_attn = AttentionBlock(False, mid_dim, self.attn_heads, self.attn_dim_head)
        self.mid_block2 = block_class(mid_dim, mid_dim)


        for i, (in_dim, out_dim) in enumerate(reversed(self.dim_tuples)):
            is_last = (i >= len(self.dim_tuples)-1)
            self.ups.append([
                block_class(in_dim + out_dim, out_dim),
                block_class(in_dim + out_dim, out_dim),
                Residual(PreNorm(AttentionBlock(True, out_dim, self.attn_heads, self.attn_dim_head))),
                Upsample1D(in_dim) if not is_last else keras_conv1d(in_dim, 3)
            ])

        self.final_resnet = block_class(2*self.dim, self.dim)
        self.final_conv = keras_conv1d(out_dim, 3)


    def call(self, x, time):
        
        x = self.init_conv(x)
        r = x.copy()

        t = self.time_mlp(time)

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
            x = keras.ops.concatenate([x, h.pop()], axis=1)
            x = block1(x, t)

            x = keras.ops.concatenate([x,h.pop()], axis=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = keras.ops.concatenate([x, r], axis=1)

        x = self.final_resnet(x, t)
        x = self.final_conv(x)
        return x

    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_multiples": self.dim_multiples,
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "resnet_norm_groups": self.resnet_norm_groups,
                "attn_dim_head": self.attn_dim_head,
                "attn_heads": self.attn_heads,
            }
        )
        return config
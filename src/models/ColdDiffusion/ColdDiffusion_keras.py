import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import einops

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
        scale_shift = None
        if (time_embedding is not None) and (self.mlp is not None):
            time_embedding = self.mlp(time_embedding)
            time_embedding = einops.rearrange(time_embedding, "b c -> b c 1")
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
            n_blocks, 
            dim, 
            attn_dim_head=32,
            attn_heads=4,
            ):
        
        super().__init__()

        self.n_blocks = n_blocks
        self.dim = dim
        self.attn_dim_head = attn_dim_head
        self.attn_heads = attn_heads


    def call(self, x, time):
        return None
    
    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "n_blocks": self.n_blocks,
                "dim": self.dim,
                "attn_dim_head": self.attn_dim_head,
                "attn_heads": self.attn_heads,
            }
        )
        return config
        

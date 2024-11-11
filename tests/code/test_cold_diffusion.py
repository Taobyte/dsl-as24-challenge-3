import numpy as np

from src.models.ColdDiffusion.cold_diffusion_model_clemens import Block, ResnetBlock, Attention, GELU, RMSNorm, NormAttentionBlock, Unet1D, SinusoidalPosEmb


def test_block():
    block = Block(8)
    input = np.zeros((32, 6000, 3))
    output = block(input)
    assert output.shape == (32, 6000, 8)

def test_block_scale_shift():
    block = Block(8)
    input = np.zeros((32, 6000, 3))
    output = block(input, scale_shift=(3, 1))
    assert output.shape == (32, 6000, 8)

def test_res_block():
    block = ResnetBlock(8, 8)
    input = np.zeros((32, 6000, 8))
    output = block(input)
    assert output.shape == (32, 6000, 8)

"""
def test_res_block_time_emb():
    block = ResnetBlock(8, time_emb_dim=(6,1))
    dim = 8
    input = np.zeros((32, 6000, dim))
    time_emb = np.zeros((32, dim * 4))
    output = block(input, time_emb)
    assert output.shape == (32, 6000, 8)
"""

def test_res_block_include_res_conv():
    block = ResnetBlock(3, 8)
    input = np.zeros((32, 6000, 3))
    output = block(input)
    assert output.shape == (32, 6000, 8)

def test_attention():
    dim = 8
    attention_block = Attention(dim)
    input = np.zeros((32,25,dim))
    output = attention_block(input)
    assert output.shape == (32, 25, dim)

def test_gelu():
    input = np.zeros((32,6000,8))
    gelu = GELU()
    output = gelu(input)
    assert output.shape == (32, 6000, 8)

def test_rmsnorm():
    dim = 8
    input = np.zeros((32,6000,dim))
    rmsnorm = RMSNorm(dim)
    output = rmsnorm(input)
    assert output.shape == (32, 6000, dim)

def test_norm_attention_block():
    dim = 8 
    input = np.zeros((32,20,dim))
    norm_attention = NormAttentionBlock(dim)
    output = norm_attention(input)
    assert output.shape == (32, 20, dim)

def test_sinusoidal_embeddings():
    dim = 8
    sin = SinusoidalPosEmb(dim)
    time = np.zeros((32,1))
    output = sin(time)
    assert output.shape == (32, 1, 8)

def test_unet():
    channels = 3
    dim = 8
    input = np.zeros((32,128,channels))
    time = np.ones(32)
    unet = Unet1D(dim, channels=channels, init_dim=dim)
    output = unet(input, time)
    assert output.shape == (32, 128, channels)




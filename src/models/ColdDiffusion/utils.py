import os

import keras
import torch
import einops

def generate_degraded_sample(eq, noise, times, T):
    """generates a diffusion sample for times, with depth T
    Args:
        - eq: tensor containing the clean earthquake signals
        - noise: tensor containin the pure noise signals
        - times: tensor containing the sampling times 
        - T: maximum sampling depth
    """
    s = 0.0008
    steps = keras.ops.arange(T)
    # cosine scheduling
    beta_0 = keras.ops.cos(s / (1 + s) * torch.pi * 0.5) ** 2
    beta_t = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_t = beta_t / beta_0
    alphas_cumprod = keras.ops.cumprod(alpha_t,axis=0)
    sqrt_alpha_cumprod = keras.ops.sqrt(alphas_cumprod)
    sqrt_one_min_alphas_cumprod = keras.ops.sqrt(1-alphas_cumprod)
    alpha_ts = sqrt_alpha_cumprod.gather(dim=0, index=times)
    one_min_alpha_ts = sqrt_one_min_alphas_cumprod.gather(dim=0, index=times)

    full_noise = eq + noise

    alpha_ts = einops.rearrange(alpha_ts, "d -> d 1 1")
    one_min_alpha_ts = einops.rearrange(one_min_alpha_ts, "d -> d 1 1")
    degraded_sample = alpha_ts * eq + one_min_alpha_ts * full_noise

    return degraded_sample
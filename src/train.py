import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import omegaconf

from models.DeepDenoiser.train import fit_deep_denoiser
from models.WaveDecompNet.train import fit_wave_decomp_net
from models.ColdDiffusion.train import fit_cold_diffusion
from models.CleanUNet.train import fit_clean_unet, fit_clean_unet_pytorch


def train_model(cfg: omegaconf.DictConfig) -> keras.Model:
    
    if cfg.model.model_name == "DeepDenoiser":
        model = fit_deep_denoiser(cfg)
    elif cfg.model.model_name == "WaveDecompNet":
        model = fit_wave_decomp_net(cfg)
    elif cfg.model.model_name == "ColdDiffusion":
        model = fit_cold_diffusion(cfg)
    elif cfg.model.model_name == "CleanUNet":
        if not cfg.model.train_pytorch:
            model = fit_clean_unet(cfg)
        else:
            model = fit_clean_unet_pytorch(cfg)
    
    return model



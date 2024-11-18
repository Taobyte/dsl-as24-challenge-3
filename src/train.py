import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import omegaconf

from models.DeepDenoiser.train import fit_deep_denoiser
from models.WaveDecompNet.train import fit_wave_decomp_net
from models.CleanUNet.train import fit_clean_unet
from models.ColdDiffusion.train import fit_cold_diffusion


def train_model(cfg: omegaconf.DictConfig) -> keras.Model:
    
    if cfg.model.model_name == "ColdDiffusion":
        model = fit_cold_diffusion(cfg)
    elif cfg.model.model_name == "DeepDenoiser":
        model = fit_deep_denoiser(cfg)
    elif cfg.model.model_name == "WaveDecompNet":
        model = fit_wave_decomp_net(cfg)
    elif cfg.model.model_name == "CleanUNet":
        model = fit_clean_unet(cfg)
    
    return model



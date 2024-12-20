import omegaconf
import torch

from src.utils import Model
from src.models.DeepDenoiser.train import fit_deep_denoiser_pytorch
from src.models.CleanUNet.train import fit_clean_unet_pytorch
from models.ColdDiffusion.train import fit_cold_diffusion


def train_model(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    if cfg.model.model_name == "DeepDenoiser":
        model = fit_deep_denoiser_pytorch(cfg)
    elif cfg.model.model_name == "CleanUNet":
        model = fit_clean_unet_pytorch(cfg, Model.CleanUNet)
    elif cfg.model.model_name == "CleanUNet2":
        model = fit_clean_unet_pytorch(cfg, Model.CleanUNet2)
    elif cfg.model.model_name == "ColdDiffusion":
        model = fit_cold_diffusion(cfg)
        raise NotImplementedError
    else:
        raise NotImplementedError

    return model

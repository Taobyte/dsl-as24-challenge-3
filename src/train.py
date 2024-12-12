import omegaconf
import torch

from src.models.DeepDenoiser.train import fit_deep_denoiser_pytorch
from src.models.CleanUNet.train import fit_clean_unet_pytorch
from src.models.CleanUNet2.train import fit_clean_unet2


def train_model(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    if cfg.model.model_name == "DeepDenoiser":
        model = fit_deep_denoiser_pytorch(cfg)
    elif cfg.model.model_name == "CleanUNet":
        model = fit_clean_unet_pytorch(cfg)
    elif cfg.model.model_name == "CleanUNet2":
        model = fit_clean_unet2(cfg)
    else:
        raise NotImplementedError

    return model

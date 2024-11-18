import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.models.DeepDenoiser.tuner import tune_model_deep_denoiser, tune_model_deep_denoiser_optuna
from src.models.CleanUNet.tuner import  tune_model_clean_unet_optuna


def tune_model(cfg: DictConfig):

    if cfg.model.model_name == "DeepDenoiser":
        best_params = tune_model_deep_denoiser(cfg)
    else: 
        raise NotImplementedError(f"{cfg.model.model_name} tuner not implemented")


def tune_model_optuna(cfg: DictConfig):

    if cfg.model.model_name == "DeepDenoiser":
        best_params = tune_model_deep_denoiser_optuna(cfg)
    elif cfg.model.model_name == "CleanUNet":
        best_params = tune_model_clean_unet_optuna(cfg)
    else: 
        raise NotImplementedError(f"{cfg.model.model_name} tuner not implemented")

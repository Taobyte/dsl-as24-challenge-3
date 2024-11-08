import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.models.DeepDenoiser.tuner import tune_model_deep_denoiser


def tune_model(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.model.model_name == "DeepDenoiser":
        best_params = tune_model_deep_denoiser(cfg)
    else: 
        raise NotImplementedError(f"{cfg.model.model_name} tuner not implemented")




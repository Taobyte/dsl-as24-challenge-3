import os
import pathlib
import hydra
import omegaconf

import torch
import numpy as np
import keras
import wandb

from models.ColdDiffusion.utils.utils_diff import create_dataloader
from models.ColdDiffusion.train_validate import train_model


def fit_cold_diffusion(cfg: omegaconf.DictConfig):
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    if cfg.user.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="earthquake denoising",
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.model.lr,
                "architecture": "ColdDiffusion",
                "dataset": "SED dataset",
                "epochs": cfg.model.epochs,
                "batch_size": cfg.model.batch_size,
                "dim": cfg.model.dim,
                "dim_multiples": cfg.model.dim_multiples,
                "num_workers": cfg.model.num_workers,
            },
        )
        wandb.run.name = "{}".format(str(output_dir).split("outputs/")[-1])

    # create the dataloaders
    tr_dl, val_dl = create_dataloader(cfg, is_noise=False)
    tr_dl_noise, val_dl_noise = create_dataloader(cfg, is_noise=True)

    min_loss = train_model(cfg, tr_dl, tr_dl_noise, val_dl, val_dl_noise)

    return min_loss

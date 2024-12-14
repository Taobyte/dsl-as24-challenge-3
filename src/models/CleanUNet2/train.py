import pathlib
import logging
import time
import hydra
import omegaconf

import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils import Model

from src.dataset import get_dataloaders_pytorch
from src.models.CleanUNet.train import train_model

logger = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_clean_specnet(cfg: omegaconf.DictConfig):
    if cfg.model.clean_specnet:
        assert cfg.model.loss == "stft"

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # Create tensorboard logger.
    tb = SummaryWriter(output_dir)

    train_dl, val_dl = get_dataloaders_pytorch(cfg, model=Model.CleanUNet2)
    net = None
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

    train_model(net, optimizer, train_dl, val_dl, cfg, tb=tb)

    return 0


def fit_clean_unet2(cfg: omegaconf.DictConfig):
    raise NotImplementedError

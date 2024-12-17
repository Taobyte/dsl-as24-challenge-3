import pathlib
import logging

import hydra
import torch
import omegaconf
import numpy as np

from src.utils import Model
from src.dataset import get_dataloaders_pytorch

from models.Butterworth.butterworth_filter import bandpass_obspy
from models.Butterworth.validate import get_metrics_butterworth
from models.DeepDenoiser.validate import (
    get_metrics_deepdenoiser,
    get_predictions_deepdenoiser,
)
from models.CleanUNet.validate import get_metrics_clean_unet, get_predictions_cleanunet
from models.ColdDiffusion.validate import get_predictions_colddiffusion


logger = logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_metrics(cfg: omegaconf.DictConfig) -> None:
    if cfg.model.model_name == Model.Butterworth.value:
        get_metrics_butterworth(cfg)
    elif cfg.model.model_name == Model.DeepDenoiser.value:
        get_metrics_deepdenoiser(cfg)
    elif cfg.model.model_name == Model.CleanUNet.value:
        get_metrics_clean_unet(cfg)
    else:
        raise NotImplementedError


def create_prediction_csv(cfg: omegaconf.DictConfig) -> None:
    """
    This function creates and saves a dataframe containing the time-domain predictions from DeepDenoiser, CleanUNet(2) & ColdDiffusion
    Args:
        cfg (omegaconf.DictConfig): the hydra config file
    Returns:
        pd.DataFrame: Dataframe with columns ['eq' , 'noise', 'noisy_eq', 'shift', 'deepdenoiser', 'cleanunet', 'colddiffusion'
    """

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    for snr in cfg.snrs:
        test_dl = get_dataloaders_pytorch(cfg, return_test=True)
        eq, noise, shift = next(iter(test_dl))

        noisy_eq = snr * eq + noise
        noisy_eq = noisy_eq.float()
        eq = snr * eq

        butterworth = np.apply_along_axis(
            lambda x: bandpass_obspy(
                x,
                freqmin=cfg.freq_range[0],
                freqmax=cfg.freq_range[1],
                df=cfg.sampling_rate,
                corners=4,
                zerophase=True,
            ),
            axis=2,
            arr=noisy_eq.numpy(),
        )

        noisy_eq = noisy_eq.to(device)

        deepdenoiser = get_predictions_deepdenoiser(noisy_eq, cfg)
        cleanunet = get_predictions_cleanunet(noisy_eq, cfg)
        colddiffusion = get_predictions_colddiffusion(noisy_eq, cfg)

        np.savez(
            output_dir / f"snr_{snr}_predictions.npz",
            eq=eq.numpy(),
            noise=noise.numpy(),
            noisy_eq=noisy_eq.cpu().numpy(),
            shift=np.array(shift),
            butterworth=butterworth,
            deepdenoiser=deepdenoiser.numpy(),
            cleanunet=cleanunet.numpy(),
            colddiffusion=colddiffusion.numpy(),
        )

import pathlib

import hydra
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

        butterworth = np.apply_along_axis(
            lambda x: bandpass_obspy(
                x,
                freqmin=cfg.butterworth_range[0],
                freqmax=cfg.butterworth_range[1],
                df=cfg.sampling_rate,
                corners=4,
                zerophase=False,
            ),
            axis=2,
            arr=noisy_eq.numpy(),
        )

        deepdenoiser = get_predictions_deepdenoiser(eq, noise, cfg)
        cleanunet = get_predictions_cleanunet(eq, noise, cfg)
        colddiffusion = get_predictions_colddiffusion(eq, noise, cfg)

        np.savez(
            output_dir / f"snr_{snr}_predictions.npz",
            eq=eq.numpy(),
            noise=noise.numpy(),
            noisy_eq=noisy_eq.numpy(),
            shift=np.array(shift),
            butterworth=butterworth,
            deepdenoiser=deepdenoiser.numpy(),
        )

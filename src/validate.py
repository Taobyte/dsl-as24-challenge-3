import tqdm
import pathlib

import hydra
import omegaconf
import torch
import numpy as np
from numpy import ndarray
import pandas as pd

from src.utils import Mode
from src.models.DeepDenoiser.dataset import (
    get_signal_noise_assoc,
    get_dataloaders_pytorch,
)
from models.Butterworth.butterworth_filter import bandpass_obspy
from models.DeepDenoiser.validate import (
    get_metrics_deepdenoiser,
    get_predictions_deepdenoiser,
)
from models.CleanUNet.validate import get_metrics_clean_unet
from metrics import cross_correlation, p_wave_onset_difference, max_amplitude_difference
from models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch


def get_metrics(
    model, assoc: list, snr: int, cfg: omegaconf.DictConfig
) -> tuple[ndarray, ndarray, ndarray]:
    if cfg.model.model_name == "DeepDenoiser":
        cross_correlations, max_amplitude_differences, p_wave_onset_differences = (
            get_metrics_deepdenoiser(model, assoc, snr, cfg, idx=0)
        )
    elif cfg.model.model_name == "CleanUNet":
        cross_correlations, max_amplitude_differences, p_wave_onset_differences = (
            get_metrics_clean_unet(model, cfg, snr, idx=0)
        )
    else:
        raise NotImplementedError(
            f"{cfg.model.model_name} get_metrics function not implemented"
        )

    return cross_correlations, max_amplitude_differences, p_wave_onset_differences


def compute_metrics(cfg: omegaconf.DictConfig) -> pd.DataFrame:
    assoc = get_signal_noise_assoc(
        cfg.user.data.signal_path, cfg.user.data.noise_path, Mode.TEST, cfg.size_testset
    )

    predictions = {
        "cc_mean": [],
        "cc_std": [],
        "max_amp_diff_mad": [],
        "max_amp_diff_std": [],
        "p_wave_mean": [],
        "p_wave_std": [],
    }

    preds_butterworth = {
        "cc_mean": [],
        "cc_std": [],
        "max_amp_diff_mad": [],
        "max_amp_diff_std": [],
        "p_wave_mean": [],
        "p_wave_std": [],
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CleanUNetPytorch(
        channels_input=3,
        channels_output=3,
        channels_H=cfg.model.channels_H,
        encoder_n_layers=cfg.model.encoder_n_layers,
        tsfm_n_layers=cfg.model.tsfm_n_layers,
        tsfm_n_head=cfg.model.tsfm_n_head,
        tsfm_d_model=cfg.model.tsfm_d_model,
        tsfm_d_inner=cfg.model.tsfm_d_inner,
    ).to(device)

    if "safetensors" in cfg.user.model_path:
        from safetensors.torch import load_file

        checkpoint = load_file(cfg.user.model_path)
    else:
        checkpoint = torch.load(cfg.user.model_path, map_location=torch.device("cpu"))

    if "model_state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print(f"running predictions for snrs {cfg.snrs}")
    for snr in tqdm.tqdm(cfg.snrs, total=len(cfg.snrs)):
        cross_correlations, max_amplitude_differences, p_wave_onset_differences = (
            get_metrics(model, assoc, snr, cfg)
        )
        # cross, SNR, max amplitude difference
        cross_correlation_mean = np.mean(cross_correlations)
        cross_correlation_std = np.std(cross_correlations)

        max_amplitude_difference_mad = np.mean(np.abs(1 - max_amplitude_differences))
        max_amplitude_difference_std = np.std(np.abs(1 - max_amplitude_differences))

        p_wave_onset_difference_mean = np.mean(p_wave_onset_differences)
        p_wave_onset_difference_std = np.std(p_wave_onset_differences)

        predictions["cc_mean"].append(cross_correlation_mean)
        predictions["cc_std"].append(cross_correlation_std)
        predictions["max_amp_diff_mad"].append(max_amplitude_difference_mad)
        predictions["max_amp_diff_std"].append(max_amplitude_difference_std)
        predictions["p_wave_mean"].append(p_wave_onset_difference_mean)
        predictions["p_wave_std"].append(p_wave_onset_difference_std)

        # butterworth
        butti = get_bandpass_results(assoc, snr=snr)

        preds_butterworth["cc_mean"].append(butti[0])
        preds_butterworth["cc_std"].append(butti[1])
        preds_butterworth["max_amp_diff_mad"].append(butti[2])
        preds_butterworth["max_amp_diff_std"].append(butti[3])
        preds_butterworth["p_wave_mean"].append(butti[4])
        preds_butterworth["p_wave_std"].append(butti[5])

    df = pd.DataFrame(predictions)
    df_butti = pd.DataFrame(preds_butterworth)
    df["snr"] = cfg.snrs
    df_butti["snr"] = cfg.snrs
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    df.to_csv(output_dir / "metrics_model.csv")
    df_butti.to_csv(output_dir / "metrics_butterworth.csv")

    return df, df_butti


def get_bandpass_results(assoc, snr, idx=0):
    """get baseline results for butterworth bandpass filter

    Args:
        - assoc: list of (event, noise, snr, shift) associations
        - snr: the signal to noise ratio at which we score
        - idx: which coordinate to score (i.e. 0=Z, 1=N, 2=E)
    Returns:
        - a dictionary with results (mean and std) for the cross-correlation,
          maximum amplitude difference (percentage) and p wave onset shift (timesteps)
    """

    test_dataset = InputSignals(assoc, Mode.TEST, snr)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    ccs = []
    amplitudes = []
    onsets = []

    # butterworth params
    freq_range = [1, 45]
    sampling_rate = 100

    for noisy_batch, eq_batch, shifts in test_dl:
        eq_batch = eq_batch[0, idx].numpy()
        filtered = bandpass_obspy(
            noisy_batch[0, idx],
            freqmin=freq_range[0],
            freqmax=freq_range[1],
            df=sampling_rate,
            corners=4,
            zerophase=False,
        )

        ccs.append(cross_correlation(eq_batch, filtered))
        amplitudes.append(max_amplitude_difference(eq_batch, filtered))
        onsets.append(p_wave_onset_difference(eq_batch, filtered, shift=shifts))

    amplitudes = np.array(amplitudes)

    # cross, SNR, max amplitude difference
    cross_correlation_mean = np.mean(ccs)
    cross_correlation_std = np.std(ccs)
    max_amplitude_difference_mad = np.mean(np.abs(1 - amplitudes))
    max_amplitude_difference_std = np.std(np.abs(1 - amplitudes))
    p_wave_onset_difference_mean = np.mean(onsets)
    p_wave_onset_difference_std = np.std(onsets)

    return [
        cross_correlation_mean,
        cross_correlation_std,
        max_amplitude_difference_mad,
        max_amplitude_difference_std,
        p_wave_onset_difference_mean,
        p_wave_onset_difference_std,
    ]


def create_prediction_csv(cfg: omegaconf.DictConfig) -> None:
    """
    This function creates and saves a dataframe containing the time-domain predictions from DeepDenoiser, CleanUNet & ColdDiffusion
    Args:
        cfg (omegaconf.DictConfig): the hydra config file
    Returns:
        pd.DataFrame: Dataframe with columns ['eq' , 'noise', 'noisy_eq', 'shift', 'deepdenoiser', 'cleanunet', 'colddiffusion'
    """

    # models = ["DeepDenoiser"]
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)
    eq, noise, shift = next(iter(test_dl))
    noisy_eq = eq + noise

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

    predictions = get_predictions_deepdenoiser(eq, noise, cfg)

    print(predictions.shape)

    np.savez(
        output_dir / "data.npz",
        eq=eq.numpy(),
        noise=noise.numpy(),
        noisy_eq=noisy_eq.numpy(),
        shift=np.array(shift),
        butterworth=butterworth,
        deepdenoiser=predictions.numpy(),
    )

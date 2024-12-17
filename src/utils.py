import logging
import pathlib
import time
from math import cos, pi

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import hydra
import omegaconf
from omegaconf import OmegaConf
from scipy.stats import wilcoxon
from enum import Enum

from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser

logger = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Model(Enum):
    Butterworth = "Butterworth"
    DeepDenoiser = "DeepDenoiser"
    ColdDiffusion = "ColdDiffusion"
    CleanUNetLSTM = "CleanUNetLSTM"
    CleanUNetTransformer = "CleanUNetTransformer"
    CleanSpecNet = "CleanSpecNet"
    CleanUNet2 = "CleanUNet2"


def test_inference_speed(cfg: omegaconf.DictConfig) -> None:
    from src.dataset import get_dataloaders_pytorch

    deepdenoiser, deepdenoiser_config = get_trained_model(cfg, Model.DeepDenoiser)
    cleanunet, cleanunet_config = get_trained_model(cfg, Model.CleanUNetTransformer)

    test_dl = get_dataloaders_pytorch(cfg, return_test=True)

    warmup_iterations = cfg.inference_speed.warmup_iterations
    timing_iterations = cfg.inference_speed.timing_iterations

    log_inference_speed(
        deepdenoiser,
        test_dl,
        warmup_iterations,
        timing_iterations,
        deepdenoiser_config.model.model_name,
    )
    log_model_size(
        deepdenoiser,
        deepdenoiser_config.model.model_name,
    )
    log_inference_speed(
        cleanunet,
        test_dl,
        warmup_iterations,
        timing_iterations,
        cleanunet_config.model.model_name,
    )
    log_model_size(cleanunet, cleanunet_config.model.model_name)


def get_trained_model(
    cfg: omegaconf.DictConfig, model_type: Model
) -> tuple[torch.nn.Module, omegaconf.DictConfig]:
    if model_type == Model.DeepDenoiser:
        config_path = cfg.user.deep_denoiser_folder + "/.hydra/config.yaml"
        config = OmegaConf.load(config_path)
        model = DeepDenoiser(**config.model.architecture).to(device)
        checkpoint = torch.load(
            cfg.user.deep_denoiser_folder + "/model.pth",
            map_location=torch.device("cpu"),
        )
    elif model_type == Model.CleanUNetTransformer:
        config_path = cfg.user.clean_unet_folder_transformer + "/.hydra/config.yaml"
        config = OmegaConf.load(config_path)
        model = CleanUNetPytorch(**config.model.architecture).to(device)
        checkpoint = torch.load(
            cfg.user.clean_unet_folder_transformer + "/model.pkl",
            map_location=torch.device("cpu"),
        )

    elif model_type == Model.CleanUNet2:
        raise NotImplementedError
    elif model_type == Model.ColdDiffusion:
        raise NotImplementedError
    else:
        raise NotImplementedError

    if "model_state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    logger.info(f"Trained model {model_type} loaded successfully.")

    return model, config


def log_model_size(net: torch.nn.Module, model_name: str) -> int:
    """
    Print the number of parameters of a network

    Args:
        net (torch.nn.Module): The PyTorch model to find parameter count.

    Returns:
        int: Number of trainable parameters of net.
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])

        logger.info(
            "{} Parameters: {:.6f}M".format(net.__class__.__name__, params / 1e6)
        )

    return params


def log_inference_speed(
    net: torch.nn.Module,
    test_dl: DataLoader,
    warmup_iterations: int,
    timing_iterations: int,
    model_name: str,
) -> float:
    """
    Measures the inference speed of a PyTorch model on a test DataLoader.

    Args:
        net (torch.nn.Module): The PyTorch model to evaluate.
        cfg (omegaconf.DictConfig): Configuration object (optional use).
        test_dl (DataLoader): DataLoader containing test data.

    Returns:
        float: Inference speed in samples per second.
    """

    net.eval()
    net.to(device)

    logger.info(f"Using device {device} for inference speed calculation.")

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iterations):
            for eq, noise, shift in test_dl:
                eq, noise = eq.float(), noise.float()
                noisy_eq = (eq + noise).to(device)
                _ = net(noisy_eq)
                break

    start_time = time.perf_counter()
    total_samples = 0

    with torch.no_grad():
        for _ in range(timing_iterations):
            for eq, noise, shift in test_dl:
                eq, noise = eq.float(), noise.float()
                noisy_eq = (eq + noise).to(device)
                _ = net(noisy_eq)
                total_samples += noisy_eq.size(0)
                break

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Calculate samples per second
    inference_speed = total_samples / elapsed_time
    logger.info(f"Inference Speed of {model_name}: {inference_speed:.2f} samples/sec")

    return inference_speed


def wilcoxon_test(cfg: omegaconf.DictConfig) -> pd.DataFrame:
    """
    Compute Wilcoxon significance test for baseline DeepDenoiser against trained model ColdDiffusion.

    Args:
        cfg (omegaconf.DictConfig): Configuration object with necessary paths and SNR values.

    Returns:
        pd.DataFrame: DataFrame storing p-values, mean of metric for DeepDenoiser / mean of metric for ColdDiffusion for each SNR and metric.
    """
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    metrics_folder = cfg.user.metrics_folder

    p_values = {}
    for snr in cfg.snrs:
        deep_df = pd.read_csv(
            metrics_folder + f"/DeepDenoiser/snr_{snr}_metrics_DeepDenoiser.csv"
        )
        model_df = pd.read_csv(
            metrics_folder + f"/ColdDiffusion/snr_{snr}_metrics_ColdDiffusion.csv"
        )
        # Drop extra column if present
        if len(model_df.columns) == 4:
            model_df = model_df.drop(model_df.columns[0], axis=1)

        # Compute mean values for both models
        mean_deep = deep_df.mean().values
        mean_model = model_df.mean().values

        # Compute Wilcoxon test p-values and store results
        p = {}
        for i, column in enumerate(deep_df.columns):
            p[column] = [
                wilcoxon(deep_df[column], model_df[column]).pvalue,
                mean_deep[i],
                mean_model[i],
            ]
        p_values[str(snr)] = p

    # Transform the results into a tidy DataFrame
    results = []
    for snr, metrics in p_values.items():
        for metric, values in metrics.items():
            results.append([snr, metric, values[0], values[1], values[2]])

    df = pd.DataFrame(
        results,
        columns=[
            "snr",
            "metric",
            "p_value",
            "deep_denoiser_mean",
            "cold_diffusion_mean",
        ],
    )

    # Save to CSV
    df.to_csv(output_dir / "p_values_ColdDiffusion.csv", index=False)

    return df


# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


####################### lr scheduler: Linear Warmup then Cosine Decay #############################

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch

# Original Copyright 2019 Kim Seonghyeon
#  MIT License (https://opensource.org/licenses/MIT)


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cosine(start, end, proportion):
    cos_val = cos(pi * proportion) + 1
    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, cur_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = cur_iter

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class LinearWarmupCosineDecay:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        iteration=0,
        divider=25,
        warmup_proportion=0.3,
        phase=("linear", "cosine"),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {"linear": anneal_linear, "cosine": anneal_cosine}

        cur_iter_phase1 = iteration
        cur_iter_phase2 = max(0, iteration - phase1)
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, cur_iter_phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, cur_iter_phase2, phase_map[phase[1]]),
        ]

        if iteration < phase1:
            self.phase = 0
        else:
            self.phase = 1

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr

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
    CleanUNet = "CleanUNet"
    CleanSpecNet = "CleanSpecNet"
    CleanUNet2 = "CleanUNet2"


def get_trained_model(cfg: omegaconf.DictConfig, model_type: Model) -> torch.nn.Module:
    if model_type == Model.DeepDenoiser:
        config_path = cfg.user.deep_denoiser_folder + "/.hydra/config.yaml"
        config = OmegaConf.load(config_path)
        model = DeepDenoiser(**config.model.architecture).to(device)
        checkpoint = torch.load(
            cfg.user.deep_denoiser_folder + "/model.pth",
            map_location=torch.device("cpu"),
        )
    elif model_type == Model.CleanUNet:
        config_path = cfg.user.clean_unet_folder + "/.hydra/config.yaml"
        config = OmegaConf.load(config_path)
        model = CleanUNetPytorch(**config.model.architecture).to(device)
        checkpoint = torch.load(
            cfg.user.clean_unet_folder + "/model.pkl",
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

    return model


def log_model_size(net: torch.nn.Module) -> int:
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


def log_inference_speed(net: torch.nn.Module, test_dl: DataLoader) -> float:
    """
    Measures the inference speed of a PyTorch model on a test DataLoader.

    Args:
        net (torch.nn.Module): The PyTorch model to evaluate.
        cfg (omegaconf.DictConfig): Configuration object (optional use).
        test_dl (DataLoader): DataLoader containing test data.

    Returns:
        float: Inference speed in samples per second.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.eval()
    net.to(device)

    logger.info(f"Using device {device} for inference speed calculation.")

    warmup_iterations = 5
    timing_iterations = 20

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iterations):
            for batch in test_dl:
                inputs = batch[0].to(device)
                _ = net(inputs)
                break

    start_time = time.perf_counter()
    total_samples = 0

    with torch.no_grad():
        for _ in range(timing_iterations):
            for batch in test_dl:
                inputs = batch[0].to(device)
                _ = net(inputs)
                total_samples += inputs.size(0)
                break

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Calculate samples per second
    inference_speed = total_samples / elapsed_time
    logger.info(f"Inference Speed: {inference_speed:.2f} samples/sec")

    return inference_speed


def wilcoxon_test(
    metrics_deepdenoiser_path: str, metrics_model_path: str, model_name: str
) -> pd.DataFrame:
    """
    Compute wilcoxon significance test for baseline DeepDenoiser against trained model

    Args:
        metrics_deepdenoiser_path (str): path to folder containing deepdenoiser metrics (cc.csv, mar.csv, pw.csv)
        metrics_model_path (str): path to folder containing model metrics (cc.csv, mar.csv, pw.csv)
        model_name (str): name of the model used in the wilcoxon significance test

    Returns:
        pd.DataFrame: DataFrame storing p-values for each snr and metric
    """
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    metrics = ["cc", "mar", "pw"]
    p_values = {}
    for metric in metrics:
        deep_df = pd.read_csv(metrics_deepdenoiser_path + f"/{metric}.csv")
        model_df = pd.read_csv(metrics_model_path + f"/{metric}.csv")
        p = {}
        for column in deep_df.columns:
            p[column] = wilcoxon(deep_df[column], model_df[column])
        p_values[metric] = p

    df = pd.DataFrame(p_values)
    df.to_csv(output_dir / f"p_values_{model_name}.csv")

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
        divider=10,
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

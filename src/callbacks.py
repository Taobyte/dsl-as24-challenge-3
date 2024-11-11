import logging
import pathlib
import omegaconf
import hydra

import nvidia_smi
import psutil
import keras
import torch
import os

from src.models.CleanUNet.validate import visualize_predictions_clean_unet

logger = logging.getLogger(__name__)


class LogCallback(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            log_msg = f"Epoch {epoch + 1}, " + ", ".join(
                [f"{k}: {v}" for k, v in logs.items()]
            )
            logger.info(log_msg)

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            log_msg = f"Batch {batch + 1}, " + ", ".join(
                [f"{k}: {v}" for k, v in logs.items()]
            )
            logger.info(log_msg)


class GPUMonitorCallback(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU Utilization: {info.gpu}% | Memory Utilization: {info.memory}%")
        nvidia_smi.nvmlShutdown()

        memory_info = psutil.virtual_memory()
        print(f"CPU utilization: {memory_info}%")


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()

        self.cfg = cfg

    def on_epoch_end(self, epoch, logs=None):
        
        visualize_predictions_clean_unet(
            self.model,
            self.cfg.user.data.signal_path,
            self.cfg.user.data.noise_path,
            self.cfg.model.signal_length,
            self.cfg.plot.n_examples,
            self.cfg.snrs,
            epoch=epoch,
        )

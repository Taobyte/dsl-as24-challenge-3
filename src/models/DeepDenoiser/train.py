import os
import pathlib

import keras
import torch
import numpy as np
import wandb
import wandb.integration.keras
import hydra
import omegaconf

from src.utils import Mode
from models.DeepDenoiser.deep_denoiser_model_2 import UNet
from models.DeepDenoiser.dataset import get_dataloaders, CSVDataset


def fit_deep_denoiser(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    print("Training Deep Denoiser")

    if cfg.user.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="earthquake denoising",
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.model.lr,
                "architecture": "DeepDenoiser UNet",
                "dataset": "SED dataset",
                "epochs": cfg.model.epochs,
                "batch_size": cfg.model.batch_size,
            },
        )
        wandb.run.name = "{}".format(os.getcwd().split('outputs/')[-1])

    # instantiate model
    model = UNet(cfg.model.n_layers, cfg.model.dropout, cfg.model.channel_base)

    # create dataloaders
    if not cfg.model.use_csv:
        train_dl, val_dl = get_dataloaders(
            cfg.user.data.signal_path, cfg.user.data.noise_path, batch_size=cfg.model.batch_size
        )
    else:
        train_dl = torch.utils.data.DataLoader(CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN), batch_size=cfg.model.batch_size)
        val_dl = torch.utils.data.DataLoader(CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION), batch_size=cfg.model.batch_size)
    
    # define learning rate scheduler and compile model
    if cfg.model.use_lr_scheduler:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            cfg.model.lr,
            decay_steps=len(train_dl),
            decay_rate=0.1,
            staircase=True)
    else:
        lr_schedule = cfg.model.lr

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath= output_dir / "checkpoints/model_at_epoch_{epoch}.keras"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.TensorBoard(
                        log_dir=output_dir / "logs",
                        histogram_freq=1,
                        write_graph=True,
                        write_images=True,
                        update_freq="epoch"
                    )
    ]

    if cfg.user.wandb:
        wandb_callbacks = [wandb.integration.keras.WandbMetricsLogger(log_freq="batch")]
        callbacks = callbacks.extend(wandb_callbacks)
    
    # build model
    sample_shape = np.zeros(
        (cfg.model.batch_size, 3, cfg.model.signal_length)
    )
    model(sample_shape)

    # plot model architecture
    model.summary() 
    keras.utils.plot_model(
        model,
        to_file= output_dir / "model_plot.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True
    )

    model.fit(train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks)

    wandb.finish()

    return model
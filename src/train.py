import os
import pathlib

os.environ["KERAS_BACKEND"] = "jax"
import keras
import wandb
import hydra
import omegaconf

from models.DeepDenoiser.deep_denoiser_model_2 import UNet
from data import get_dataloaders


def train_model(cfg: omegaconf.DictConfig) -> keras.Model:

    model = fit_deep_denoiser(cfg)
    
    return model


def fit_deep_denoiser(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

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

    model = UNet()

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.model.lr),
        # metrics=[keras.metrics.BinaryCrossentropy()]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath= output_dir / "model_at_epoch_{epoch}.keras"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    if cfg.user.wandb:
        wandb_callbacks = [wandb.integration.keras.WandbMetricsLogger(log_freq="batch")]
        callbacks = callbacks.extend(wandb_callbacks)

    train_dl, val_dl = get_dataloaders(
        cfg.user.data.signal_path, cfg.user.data.noise_path, cfg.model.batch_size
    )

    model.fit(train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks)

    # model.evaluate(val_dl, batch_size=32, verbose=2, steps=1)

    wandb.finish()

    return model
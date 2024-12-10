import os
import pathlib
import logging
import time

import keras
import torch
import numpy as np
import wandb
import wandb.integration.keras
import hydra
import omegaconf
from tqdm import tqdm

from src.utils import Mode, log_model_size
from torch.utils.tensorboard import SummaryWriter

from models.DeepDenoiser.deep_denoiser_model_2 import UNet
from models.DeepDenoiser.dataset import (
    get_dataloaders,
    CSVDataset,
    get_dataloaders_pytorch,
)

from models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser

logger = logging.getLogger()


def fit_deep_denoiser(cfg: omegaconf.DictConfig) -> keras.Model:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

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
        wandb.run.name = "{}".format(os.getcwd().split("outputs/")[-1])

    # instantiate model
    model = UNet(cfg.model.n_layers, cfg.model.dropout, cfg.model.channel_base)

    # create dataloaders
    if not cfg.model.use_csv:
        train_dl, val_dl = get_dataloaders(
            cfg.user.data.signal_path,
            cfg.user.data.noise_path,
            batch_size=cfg.model.batch_size,
        )
    else:
        train_dl = torch.utils.data.DataLoader(
            CSVDataset(
                cfg.user.data.csv_path,
                cfg.model.signal_length,
                cfg.model.snr_lower,
                cfg.model.snr_upper,
                Mode.TRAIN,
            ),
            batch_size=cfg.model.batch_size,
        )
        val_dl = torch.utils.data.DataLoader(
            CSVDataset(
                cfg.user.data.csv_path,
                cfg.model.signal_length,
                cfg.model.snr_lower,
                cfg.model.snr_upper,
                Mode.VALIDATION,
            ),
            batch_size=cfg.model.batch_size,
        )

    # define learning rate scheduler and compile model
    if cfg.model.use_lr_scheduler:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            cfg.model.lr, decay_steps=len(train_dl), decay_rate=0.1, staircase=True
        )
    else:
        lr_schedule = cfg.model.lr

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "checkpoints/model_at_epoch_{epoch}.keras"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.TensorBoard(
            log_dir=output_dir / "logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
    ]

    if cfg.user.wandb:
        wandb_callbacks = [wandb.integration.keras.WandbMetricsLogger(log_freq="batch")]
        callbacks = callbacks.extend(wandb_callbacks)

    # build model
    sample_shape = np.zeros((cfg.model.batch_size, 3, cfg.model.signal_length))
    model(sample_shape)

    # plot model architecture
    model.summary()
    keras.utils.plot_model(
        model,
        to_file=output_dir / "model_plot.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )

    model.fit(
        train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks
    )

    wandb.finish()

    return model


def fit_deep_denoiser_pytorch(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    os.makedirs(output_dir / "checkpoints", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    tb = SummaryWriter(output_dir)

    model = DeepDenoiser(**cfg.model.architecture).to(device)
    log_model_size(model)

    optimizer = torch.optim.AdamW(model.parameters(), cfg.model.lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    train_dl, val_dl = get_dataloaders_pytorch(cfg, subset=cfg.model.subset)
    if cfg.model.subset:
        logger.info(f"using {cfg.model.subset} data points for training")

    logger.info("Start training DeepDenoiser")
    best_val_loss = float("inf")
    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        train_loss = 0.0
        time0 = time.time()
        for noisy_eq, gt_mask in train_dl:
            optimizer.zero_grad()
            noisy_eq = noisy_eq.to(device)
            gt_mask = gt_mask.to(device)
            predicted_mask = model(noisy_eq)
            loss = bce_loss(predicted_mask, gt_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.model.clip_norm
            )
            tb.add_scalar("gradient_norm", grad_norm, epoch)
        time1 = time.time()
        train_loss /= len(train_dl)
        tb.add_scalar("train_loss", train_loss, epoch)
        logger.info(
            f"Epoch {epoch} |  train_loss: {train_loss} | grad_norm: {grad_norm} | epoch_time: {time1 - time0}"
        )

        # validation
        if epoch % cfg.model.val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for noisy_eq, gt_mask in tqdm(val_dl, total=len(val_dl)):
                    noisy_eq = noisy_eq.to(device)
                    gt_mask = gt_mask.to(device)
                    predicted_mask = model(noisy_eq)
                    loss = bce_loss(predicted_mask, gt_mask)
                    val_loss += loss.item()
                val_loss /= len(val_dl)
                tb.add_scalar("validation_loss", val_loss, epoch)

                logger.info(f"Epoch {epoch} |  val_loss: {val_loss}")

                if val_loss < best_val_loss:
                    torch.save(
                        model.state_dict(),
                        output_dir / f"checkpoints/epoch_{epoch}.pth",
                    )
                    best_val_loss = val_loss
                    logger.info(
                        f"New model saved at epoch: {epoch} | best_val_loss: {best_val_loss}"
                    )

    torch.save(model, output_dir)

    return model

import pathlib
import hydra
import omegaconf 

import torch
import numpy as np
import keras

from src.utils import Mode
from src.models.CleanUNet.dataset import CleanUNetDatasetCSV
from src.models.ColdDiffusion.cold_diffusion_model_clemens import Unet1D
from src.models.ColdDiffusion.dataset import ColdDiffusionDataset
from src.models.CleanUNet.utils import CleanUNetLoss

def fit_cold_diffusion(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    model = Unet1D(
        cfg.model.dim,
        init_dim=cfg.model.dim,
        dim_mults=tuple(cfg.model.dim_mults),
        channels=cfg.model.channels,
    )

    # build model
    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels)
    )
    model(sample_shape)

    if cfg.model.loss == 'stft':
        loss = CleanUNetLoss(signal_length=cfg.model.signal_length)
    elif cfg.model.loss == 'mae':
        loss = keras.losses.MeanAbsoluteError()
    elif cfg.model.loss == 'mse':
        loss = keras.losses.MeanSquaredError()
    else:
        print(f"loss {cfg.model.loss} is not supported")

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.model.lr),
    )

    # build model
    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels)
    )
    model(sample_shape)

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
    if not cfg.model.use_csv:
        train_dataset = ColdDiffusionDataset(
            cfg.user.data.signal_path + "/train/",
            cfg.user.data.noise_path + "/train/",
            cfg.model.signal_length,
        )
        val_dataset = ColdDiffusionDataset(
            cfg.user.data.signal_path + "/validation/",
            cfg.user.data.noise_path + "/validation/",
            cfg.model.signal_length,
        )
    else:
        train_dataset = CleanUNetDatasetCSV(
            cfg.user.data.csv_path,
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            cfg.model.event_shift_start,
            Mode.TRAIN
        )
        val_dataset = CleanUNetDatasetCSV(
            cfg.user.data.csv_path,
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            cfg.model.event_shift_start,
            Mode.VALIDATION
        )

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size
    )
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)

    model.summary() 

    model.fit(
        train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks
    )

    return model

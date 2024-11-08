import pathlib
import hydra
import omegaconf 

import torch
import jax
import numpy as np
import keras

from src.metrics import AmpMetric
from src.utils import Mode
from src.models.CleanUNet.clean_unet_model import CleanUNet, CleanUNetLoss
from src.models.CleanUNet.dataset import CleanUNetDataset, CleanUNetDatasetCSV


def fit_clean_unet(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    model = CleanUNet(cfg.model.channels_input, 
                      cfg.model.channels_output, 
                      cfg.model.signal_length, 
                      cfg.model.channels_H,
                      cfg.model.max_H,
                      cfg.model.encoder_n_layers,
                      cfg.model.kernel_size,
                      cfg.model.stride,
                      cfg.model.tsfm_n_layers,
                      cfg.model.tsfm_n_head,
                      cfg.model.tsfm_d_model,
                      cfg.model.tsfm_d_inner)

    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels_input)
    )
    model(sample_shape)

    # metrics = [AmpMetric()]
    metrics = []

    if cfg.model.loss == "stft":
        loss = CleanUNetLoss()
    elif cfg.model.loss == "mae":
        loss = keras.losses.MeanAbsoluteError()
    else:
        print(f"loss function : {cfg.model.loss} not supported")

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.model.lr),
        metrics=metrics
    )

    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels_input)
    )
    model(sample_shape)

    # Total parameters
    total_params = model.count_params()

    # Assuming 32-bit floats (4 bytes per parameter)
    memory_size_in_bytes = total_params * 4

    # Convert to MB
    memory_size_in_MB = memory_size_in_bytes / (1024 ** 2)
    print(f"Model size: {memory_size_in_MB:.2f} MB")

    model.summary()

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
        train_dataset = CleanUNetDataset(
            cfg.user.data.signal_path + "/train/",
            cfg.user.data.noise_path + "/train/",
            cfg.model.signal_length
        )
        val_dataset = CleanUNetDataset(
            cfg.user.data.signal_path + "/validation/",
            cfg.user.data.noise_path + "/validation/",
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            cfg.model.event_shift_start
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

    jax.profiler.start_trace(output_dir)

    model.fit(
        train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks
    )

    jax.profiler.stop_trace()

    return model
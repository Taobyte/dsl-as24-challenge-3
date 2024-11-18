import pathlib
import hydra
import omegaconf 

import torch
import keras

from callbacks import GPUMonitorCallback, LogCallback
from utils import Mode
from metrics import AmpMetric
from models.WaveDecompNet.wave_decomp_net import UNet1D, WaveDecompLoss
from models.WaveDecompNet.dataset import WaveDecompNetDataset, WaveDecompNetDatasetCSV


def fit_wave_decomp_net(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    model = UNet1D(cfg.model.n_layers, cfg.model.dropout, cfg.model.channel_base, cfg.model.n_lstm_layers)

    model.compile(
        loss=WaveDecompLoss(cfg.model.signal_weight),
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.model.lr),
    )
    
    # GPUMonitorCallback()
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
                    ),
    ]
    if cfg.model.use_csv:
        train_dataset = WaveDecompNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
        val_dataset = WaveDecompNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)
    else:
        train_dataset = WaveDecompNetDataset(cfg.user.data.signal_path + "/train/", cfg.user.data.noise_path + "/train/", cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
        val_dataset = WaveDecompNetDataset(cfg.user.data.signal_path + "/validation/", cfg.user.data.noise_path + "/validation/", cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.batch_size)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)

    model.fit(train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks)

    return model
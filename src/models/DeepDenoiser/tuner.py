import keras_tuner
import keras 

import hydra
from omegaconf import DictConfig
from pathlib import Path

from models.DeepDenoiser.deep_denoiser_model_2 import UNet
from src.models.DeepDenoiser.dataset import get_dataloaders

def build_model(hp):
    channel_base = hp.Choice('channel_base', [2, 4, 8, 16, 32])
    dropout = hp.Float('dropout', min_value=0, max_value=1)
    model = UNet(channel_base=channel_base, dropout=dropout)

    learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=0.01)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

    return model


def tune_model_deep_denoiser(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=5,
        directory=output_dir)
    
    tuner.search_space_summary()

    train_dl, val_dl = get_dataloaders(cfg.data.signal_path, cfg.data.noise_path)
    
    tuner.search(train_dl, epochs=5, validation_data=val_dl)
    best_hypers = tuner.get_best_hyperparameters()

    tuner.results_summary()

    return best_hypers

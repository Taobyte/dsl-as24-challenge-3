import hydra
from omegaconf import DictConfig
from pathlib import Path

import keras_tuner
import keras 
import torch
import optuna

from src.utils import Mode
from models.DeepDenoiser.deep_denoiser_model_2 import UNet
from src.models.DeepDenoiser.dataset import get_dataloaders, CSVDataset

# ======================= KERAS TUNER ===================================================

def build_model(hp):

    channel_base = hp.Choice('channel_base', [2, 4, 8, 16, 32])
    n_layers = hp.Choice('n_layers', [2,3,4,5,6])
    dropout = hp.Float('dropout', min_value=0, max_value=1)
    model = UNet(n_layers=n_layers, channel_base=channel_base, dropout=dropout)

    learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=0.1)

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
        max_trials=cfg.n_trials,
        directory=output_dir)
    
    tuner.search_space_summary()

    # train_dl, val_dl = get_dataloaders(cfg.data.signal_path, cfg.data.noise_path)
    train_dl = torch.utils.data.DataLoader(CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN), batch_size=cfg.model.batch_size)
    val_dl = torch.utils.data.DataLoader(CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION), batch_size=cfg.model.batch_size)

    # define callbacks 
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001),
        keras.callbacks.TerminateOnNaN()
    ]

    tuner.search(train_dl, epochs=5, validation_data=val_dl, callbacks=callbacks)
    best_hypers = tuner.get_best_hyperparameters()

    tuner.results_summary()

    return best_hypers


# =================================== OPTUNA ===================================================

def tune_model_deep_denoiser_optuna(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
    val_dataset = CSVDataset(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)
    
    def objective(trial):
        # Define the hyperparameters for tuning
        channel_base = trial.suggest_categorical('channel_base', [2, 4, 8, 16, 32])
        n_layers = trial.suggest_int('n_layers', 2, 6)  # Define as an integer range
        dropout = trial.suggest_float('dropout', 0, 1)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # Add batch size options here
        epochs = trial.suggest_int('epochs', 3, 10)
        
        # Build the model with the suggested hyperparameters
        model = UNet(n_layers=n_layers, channel_base=channel_base, dropout=dropout)
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # Train the model
        history = model.fit(train_dl, epochs=epochs, validation_data=val_dl, verbose=0)

        # Use the last validation loss as the objective to minimize
        val_loss = history.history['val_loss'][-1]

        return val_loss

        # Configure Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)  # Define the number of trials

    # Retrieve and summarize best hyperparameters
    best_hypers = study.best_params
    print("Best hyperparameters:", best_hypers)

    study.trials_dataframe().to_csv(output_dir / "optuna_results.csv")

    return best_hypers

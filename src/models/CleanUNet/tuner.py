import hydra
from omegaconf import DictConfig
from pathlib import Path

import keras_tuner
import keras 
import torch
import optuna

from src.utils import Mode
from models.CleanUNet.clean_unet_model import CleanUNet
from src.models.CleanUNet.dataset import CleanUNetDatasetCSV

# ======================= KERAS TUNER ===================================================

def build_model(hp):
    raise NotImplementedError("This method needs to be implemented.")

def tune_model_deep_denoiser(cfg: DictConfig):
    raise NotImplementedError("This method needs to be implemented.")



# =================================== OPTUNA ===================================================

def tune_model_deep_denoiser_optuna(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
    val_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)
    
    def objective(trial):
        # Define the hyperparameters for tuning

        dropout = trial.suggest_float('dropout', 0, 1)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # Add batch size options here
        epochs = trial.suggest_int('epochs', 3, 10)
        
        # Build the model with the suggested hyperparameters
        model = CleanUNet(dropout=dropout)
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

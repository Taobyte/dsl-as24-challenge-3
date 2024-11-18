import hydra
from omegaconf import DictConfig
from pathlib import Path

import keras_tuner
import keras 
import torch
import optuna

from src.utils import Mode
from models.CleanUNet.clean_unet_model import CleanUNet
from models.CleanUNet.clean_unet2_model import baseline_unet
from src.models.CleanUNet.utils import CleanUNetLoss
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

        dropout = trial.suggest_float('dropout', 0, 1)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]) 
        epochs = trial.suggest_int('epochs', 3, 50)
        
        # Build the model with the suggested hyperparameters
        if not cfg.model.use_baseline:
            model = CleanUNet(dropout=dropout)
        else:
            channel_base = trial.suggest_categorical("channel_base", [4, 8, 16])
            model = baseline_unet(cfg.model.signal_length, cfg.model.channel_dims, channel_base)
        
        choices = [8, 16, 32, 64, 128, 256, 512]
        frame_lengths = trial.suggest_categorical("frame_lengths", choices) 
        frame_steps = trial.suggest_categorical("frame_steps", choices) 
        fft_sizes = trial.suggest_categorical("fft_sizes", [fft_length for fft_length in choices if fft_length >= frame_lengths]) 

        model.compile(
            loss= CleanUNetLoss(cfg.model.signal_length, frame_lengths, frame_steps, fft_sizes),
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate)
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

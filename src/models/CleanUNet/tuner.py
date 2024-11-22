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
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.CleanUNet.train import train_model

# ======================= KERAS TUNER ===================================================

def build_model(hp):
    raise NotImplementedError("This method needs to be implemented.")

def tune_model_deep_denoiser(cfg: DictConfig):
    raise NotImplementedError("This method needs to be implemented.")



# =================================== OPTUNA ===================================================

def tune_model_clean_unet_optuna(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
    val_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)
    
    choices = [4, 8, 16, 32, 64, 128, 256, 512]
    zipped = []
    for frame_l in choices:
        for frame_s in choices:
            for fft_l in choices:
                if fft_l >= frame_l:
                    zipped.append((frame_l, frame_s, fft_l))

    def objective(trial):

        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        epochs = trial.suggest_int('epochs', 3, 50)
        frame_lengths, frame_steps, fft_sizes = trial.suggest_categorical("stft_params", zipped) 

        print("Next Trial")
        print(f"Learning Rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Frame Length: {frame_lengths}")
        print(f"Frame Step: {frame_steps}")
        print(f"FFT Size: {frame_lengths}")
        
        # Build the model with the suggested hyperparameters
        if not cfg.model.use_baseline:
            model = CleanUNet(dropout=cfg.model.dropout)
        else:
            # channel_base = trial.suggest_categorical("channel_base", [4, 8, 16])
            model = baseline_unet(cfg.model.signal_length, cfg.model.channel_dims, cfg.model.channel_base)
        

        model.compile(
            loss= CleanUNetLoss(cfg.model.signal_length, [frame_lengths], [frame_steps], [fft_sizes]),
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate)
        )
        
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.batch_size)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)

        callbacks = [keras.callbacks.TerminateOnNaN(),
                     keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.model.patience)
                     ]

        # Train the model
        history = model.fit(train_dl, epochs=epochs, validation_data=val_dl, verbose=0, callbacks=callbacks)

        # Use the last validation loss as the objective to minimize
        val_loss = history.history['val_loss'][-1]

        print(f"Trial finished with validation loss: {val_loss}")

        return val_loss

    # Configure Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)  # Define the number of trials

    # Retrieve and summarize best hyperparameters
    best_hypers = study.best_params
    print("Best hyperparameters:", best_hypers)

    study.trials_dataframe().to_csv(output_dir / "optuna_results.csv")

    return best_hypers


def tune_model_clean_unet_pytorch_optuna(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.TRAIN)
    val_dataset = CleanUNetDatasetCSV(cfg.user.data.csv_path, cfg.model.signal_length, cfg.model.snr_lower, cfg.model.snr_upper, Mode.VALIDATION)
    
    def objective(trial):
        
        # encoder params
        encoder_n_layers = trial.suggest_int("encoder_n_layers", 3, 8) 
        channels_H = trial.suggest_catgorical("channels_H", [4, 8, 16, 32, 64])

        # transformer params
        tsfm_n_layers = trial.suggest_int("tsfm_n_layers", 1, 4)
        tsfm_n_head = trial.suggest_categorical("tsfm_n_head", [2, 4, 8])
        tsfm_d_model = trial.suggest_categorical("tsfm_d_model", [32, 64, 128, 256])
        tsfm_d_inner = trial.suggest_categorical("tsfm_d_model", [32, 64, 128, 256, 512])

        print("Next Trial")
        print(f"encoder_n_layers: {encoder_n_layers}")
        print(f"channels_H: {channels_H}")
        print(f"tsfm_n_layers: {tsfm_n_layers}")
        print(f"tsfm_n_head: {tsfm_n_head}")
        print(f"tsfm_d_model: {tsfm_d_model}")
        print(f"tsfm_d_inner: {tsfm_d_inner}")
        
        net = CleanUNetPytorch(channels_input=3, 
                                 channels_output=3,
                                 channels_H=channels_H,
                                 encoder_n_layers=encoder_n_layers,
                                 tsfm_n_layers=tsfm_n_layers,
                                 tsfm_n_head=tsfm_n_head,
                                 tsfm_d_model=tsfm_d_model,
                                 tsfm_d_inner=tsfm_d_inner
                                 )
        
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.batch_size)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

        val_loss = train_model(net, optimizer,train_dl,val_dl)

        print(f"Trial finished with validation loss: {val_loss}")

        return val_loss

    # Configure Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)  # Define the number of trials

    # Retrieve and summarize best hyperparameters
    best_hypers = study.best_params
    print("Best hyperparameters:", best_hypers)

    study.trials_dataframe().to_csv(output_dir / "optuna_results.csv")

    return best_hypers

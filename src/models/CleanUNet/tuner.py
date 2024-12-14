import logging
import hydra
import math
from omegaconf import DictConfig
from pathlib import Path


import keras_tuner
import keras
import torch
import optuna

<<<<<<< HEAD
from utils import Mode
from models.CleanUNet.clean_unet_model import CleanUNet
from models.CleanUNet.dataset import CleanUNetDatasetCSV
=======

from src.utils import Mode
from src.models.CleanUNet.clean_unet_model import CleanUNet
from src.models.CleanUNet.clean_unet2_model import baseline_unet
from src.models.CleanUNet.utils import CleanUNetLoss
from src.models.CleanUNet.dataset import CleanUNetDatasetCSV
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.CleanUNet.train import train_model


logger = logging.getLogger()
>>>>>>> 579f5c53c4137155e4d5229184af147fa9d35de3

# ======================= KERAS TUNER ===================================================


def build_model(hp):
    raise NotImplementedError("This method needs to be implemented.")


def tune_model_deep_denoiser(cfg: DictConfig):
    raise NotImplementedError("This method needs to be implemented.")


# =================================== OPTUNA ===================================================


def tune_model_clean_unet_optuna(cfg: DictConfig):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.TRAIN,
    )
    val_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.VALIDATION,
    )

    choices = [4, 8, 16, 32, 64, 128, 256, 512]
    zipped = []
    for frame_l in choices:
        for frame_s in choices:
            for fft_l in choices:
                if fft_l >= frame_l:
                    zipped.append((frame_l, frame_s, fft_l))

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        epochs = trial.suggest_int("epochs", 3, 50)
        frame_lengths, frame_steps, fft_sizes = trial.suggest_categorical(
            "stft_params", zipped
        )

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
            model = baseline_unet(
                cfg.model.signal_length, cfg.model.channel_dims, cfg.model.channel_base
            )

        model.compile(
            loss=CleanUNetLoss(
                cfg.model.signal_length, [frame_lengths], [frame_steps], [fft_sizes]
            ),
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        )

        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.model.batch_size
        )
        val_dl = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.model.batch_size
        )

        callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.model.patience
            ),
        ]

        # Train the model
        history = model.fit(
            train_dl,
            epochs=epochs,
            validation_data=val_dl,
            verbose=0,
            callbacks=callbacks,
        )

        # Use the last validation loss as the objective to minimize
        val_loss = history.history["val_loss"][-1]

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.TRAIN,
        data_format="channel_first",
    )
    val_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.VALIDATION,
        data_format="channel_first",
    )

    def objective(trial):
        # encoder params
        encoder_n_layers = trial.suggest_int("encoder_n_layers", 3, 8)
        channels_H = trial.suggest_categorical("channels_H", [4, 8, 16, 32, 64])

        # transformer params
        tsfm_n_layers = trial.suggest_int("tsfm_n_layers", 1, 4)
        tsfm_n_head = trial.suggest_categorical("tsfm_n_head", [2, 4, 8])
        tsfm_d_model = trial.suggest_categorical("tsfm_d_model", [32, 64, 128, 256])
        tsfm_d_inner = trial.suggest_categorical(
            "tsfm_d_inner", [32, 64, 128, 256, 512]
        )

        logger.info("Next Trial")
        logger.info(f"encoder_n_layers: {encoder_n_layers}")
        logger.info(f"channels_H: {channels_H}")
        logger.info(f"tsfm_n_layers: {tsfm_n_layers}")
        logger.info(f"tsfm_n_head: {tsfm_n_head}")
        logger.info(f"tsfm_d_model: {tsfm_d_model}")
        logger.info(f"tsfm_d_inner: {tsfm_d_inner}")

        net = CleanUNetPytorch(
            channels_input=3,
            channels_output=3,
            channels_H=channels_H,
            encoder_n_layers=encoder_n_layers,
            tsfm_n_layers=tsfm_n_layers,
            tsfm_n_head=tsfm_n_head,
            tsfm_d_model=tsfm_d_model,
            tsfm_d_inner=tsfm_d_inner,
            kernel_size=cfg.model.kernel_size,
            stride=cfg.model.stride,
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

        val_loss = train_model(net, optimizer, train_dataset, val_dataset, cfg)

        logger.info(f"Trial finished with validation loss: {val_loss}")

        return val_loss

    # Configure Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)  # Define the number of trials

    # Retrieve and summarize best hyperparameters
    best_hypers = study.best_params
    logger.info("Best hyperparameters:", best_hypers)

    study.trials_dataframe().to_csv(output_dir / "optuna_results.csv")

    return best_hypers


def tune_stft_loss_clean_unet_pytorch_optuna(cfg: DictConfig):
    assert cfg.model.use_metrics

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # dataset
    train_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.TRAIN,
        data_format="channel_first",
    )
    val_dataset = CleanUNetDatasetCSV(
        cfg.user.data.csv_path,
        cfg.model.signal_length,
        cfg.model.snr_lower,
        cfg.model.snr_upper,
        Mode.TEST,
        data_format="channel_first",
    )

    def objective(trial):
        # Loss tradeoff
        stft_lambda = trial.suggest_float("stft_lambda", 0.0, 1.0)
        sc_lambda = trial.suggest_float("sc_lambda", 0.0, 1.0)
        mag_lambda = trial.suggest_float("mag_lambda", 0.0, 1.0)

        """
        TODO: tune STFT parameters
        tsfm_n_layers = trial.suggest_int("tsfm_n_layers", 1, 4)
        tsfm_n_head = trial.suggest_categorical("tsfm_n_head", [2, 4, 8])
        tsfm_d_model = trial.suggest_categorical("tsfm_d_model", [32, 64, 128, 256])
        tsfm_d_inner = trial.suggest_categorical("tsfm_d_inner", [32, 64, 128, 256, 512])
        """
        logger.info("Next Trial")
        logger.info(f"stft_lambda: {stft_lambda}")
        logger.info(f"sc_lambda: {sc_lambda}")
        logger.info(f"mag_lambda: {mag_lambda}")

        net = CleanUNetPytorch(
            channels_input=3,
            channels_output=3,
            channels_H=cfg.model.channels_H,
            encoder_n_layers=cfg.model.encoder_n_layers,
            tsfm_n_layers=cfg.model.tsfm_n_layers,
            tsfm_n_head=cfg.model.tsfm_n_head,
            tsfm_d_model=cfg.model.tsfm_d_model,
            tsfm_d_inner=cfg.model.tsfm_d_inner,
            kernel_size=cfg.model.kernel_size,
            stride=cfg.model.stride,
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

        val_loss, metrics = train_model(
            net,
            optimizer,
            train_dataset,
            val_dataset,
            cfg,
            sc_lambda=sc_lambda,
            mag_lambda=mag_lambda,
            stft_lambda=stft_lambda,
        )

        # we want to minimize a loss function based on the metrics
        cc_loss = (1 - metrics["cc_mean"]) + metrics["cc_std"]
        # ma_loss = math.abs(1 - metrics['ma_mean']) + metrics['ma_std']
        pw_loss = (metrics["pw_mean"] + metrics["pw_std"]) / 2000

        metric_loss = 0.5 * cc_loss + 0.5 * pw_loss

        logger.info(f"Trial finished with validation loss: {val_loss}")
        logger.info(
            f"Weighted metrics loss: {metric_loss} | cc_loss: {cc_loss} | pw_loss:{pw_loss}"
        )

        return metric_loss

    # Configure Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)  # Define the number of trials

    # Retrieve and summarize best hyperparameters
    best_hypers = study.best_params
    logger.info("Best hyperparameters:", best_hypers)

    study.trials_dataframe().to_csv(output_dir / "optuna_results.csv")

    return best_hypers

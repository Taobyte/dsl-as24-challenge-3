import pathlib
import logging
import time
import hydra
import omegaconf

import torch
import jax
import numpy as np
import keras

from src.metrics import AmpMetric
from src.utils import Mode
from src.callbacks import VisualizeCallback
from src.models.CleanUNet.clean_unet_model import CleanUNet
from src.models.CleanUNet.utils import CleanUNetLoss
from src.models.CleanUNet.dataset import CleanUNetDataset, CleanUNetDatasetCSV
from src.models.CleanUNet.validate import visualize_predictions_clean_unet
from src.models.CleanUNet.clean_unet2_model import baseline_model, baseline_unet

from torch.utils.tensorboard import SummaryWriter
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.CleanUNet.stft_loss import MultiResolutionSTFTLoss, STFTLoss


logger = logging.getLogger() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fit_clean_unet(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if not cfg.model.use_baseline:

        model = CleanUNet(
            cfg.model.channels_input,
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
            cfg.model.tsfm_d_inner,
            cfg.model.bottleneck,
            cfg.model.use_raglu,
            cfg.model.kernel_sizes,
        )
    else:
        model = baseline_unet(
            cfg.model.signal_length,
            cfg.model.channel_dims,
            cfg.model.channel_base,
            cfg.model.kernel_size,
        )

    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels_input)
    )
    model(sample_shape)

    # metrics = [AmpMetric()]
    metrics = []

    print(
        f"type of omegaconf list {cfg.model.frame_lengths} is {type(cfg.model.frame_lengths)}"
    )
    print(type(list(cfg.model.frame_lengths)))

    if cfg.model.loss == "stft":
        loss = CleanUNetLoss(
            cfg.model.signal_length,
            list(cfg.model.frame_lengths),
            list(cfg.model.frame_steps),
            list(cfg.model.fft_sizes),
        )
    elif cfg.model.loss == "mae":
        loss = keras.losses.MeanAbsoluteError()
    elif cfg.model.loss == "mse":
        loss = keras.losses.MeanSquaredError()
    else:
        print(f"loss function : {cfg.model.loss} not supported")

    if cfg.model.lr_schedule:
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            cfg.model.lr,
            cfg.model.decay_steps,
            alpha=0.0,
            name="CosineDecay",
            warmup_target=cfg.model.warmup_target,
            warmup_steps=cfg.model.warmup_steps,
        )
    else:
        lr_schedule = cfg.model.lr

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, clipnorm=cfg.model.clipnorm
        ),
        metrics=metrics,
    )

    sample_shape = np.zeros(
        (cfg.model.batch_size, cfg.model.signal_length, cfg.model.channels_input)
    )
    model(sample_shape)

    model.summary()

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=output_dir / "logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    if cfg.model.reduce_lr_plateau:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.model.factor,
                patience=cfg.model.plateau_patience,
                min_lr=cfg.model.min_lr,
            )
        )

    if cfg.model.log_checkpoints:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=output_dir / "checkpoints/model_at_epoch_{epoch}.keras"
            )
        )

    if cfg.plot.visualization:
        callbacks.append(VisualizeCallback(cfg))

    if cfg.model.patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.model.patience
            )
        )

    if not cfg.model.use_csv:
        train_dataset = CleanUNetDataset(
            cfg.user.data.signal_path + "/train/",
            cfg.user.data.noise_path + "/train/",
            cfg.model.signal_length,
        )
        val_dataset = CleanUNetDataset(
            cfg.user.data.signal_path + "/validation/",
            cfg.user.data.noise_path + "/validation/",
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
        )
    else:
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

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size
    )
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)

    model.fit(
        train_dl, epochs=cfg.model.epochs, validation_data=val_dl, callbacks=callbacks
    )

    if cfg.plot.visualization:
        visualize_predictions_clean_unet(
            model,
            cfg.user.data.signal_path,
            cfg.user.data.noise_path,
            cfg.model.signal_length,
            cfg.plot.n_examples,
            cfg.snrs,
        )

    return model


def fit_clean_unet_pytorch(cfg: omegaconf.DictConfig):
    
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # Create tensorboard logger.
    tb = SummaryWriter(output_dir)
    if not cfg.model.load_dummy:
        # load training data
        train_dataset = CleanUNetDatasetCSV(
            cfg.user.data.csv_path,
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            Mode.TRAIN,
            data_format="channel_first"
        )
        val_dataset = CleanUNetDatasetCSV(
            cfg.user.data.csv_path,
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            Mode.VALIDATION,
            data_format="channel_first"
        )
    else: 
        inps = torch.randn((32, 3, 6120))
        tgts = torch.randn((32, 3, 6120))
        train_dataset = torch.utils.data.TensorDataset(inps, tgts)
        val_dataset = train_dataset
    
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size
    )
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.batch_size)
    print("Data loaded")

    # predefine model
    net = CleanUNetPytorch(
        channels_input=3,
        channels_output=3,
        channels_H=cfg.model.channels_H,
        encoder_n_layers=cfg.model.encoder_n_layers,
        tsfm_n_layers=cfg.model.tsfm_n_layers,
        tsfm_n_head=cfg.model.tsfm_n_head,
        tsfm_d_model=cfg.model.tsfm_d_model,
        tsfm_d_inner=cfg.model.tsfm_d_inner,
    ).to(device)
    
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

    train_model()

    return 0


def train_model(net, optimizer, train_dl, val_dl, cfg, tb=None):

    # loss function
    time_domain_loss = torch.nn.L1Loss()
    if cfg.model.loss == "stft":
        stft_loss = MultiResolutionSTFTLoss(cfg.model.fft_sizes, cfg.model.frame_lengths, cfg.model.frame_steps, sc_lambda=cfg.model.sc_lambda, mag_lambda=cfg.model.mag_lambda).to(device)
    else:
        stft_loss = None

    def get_loss(time_domain_loss, stft_loss, denoised_audio, clean_audio):
        loss = time_domain_loss(denoised_audio, clean_audio)
        if stft_loss:
            spec_loss, mag_loss = stft_loss(denoised_audio, clean_audio)
            loss += (spec_loss + mag_loss) * cfg.model.stft_lambda
        return loss
    
    # training

    best_val_loss = float('inf')
    best_epoch = 0
    patience = cfg.model.patience  # Number of epochs to wait for improvement
    stop_training = False

    time0 = time.time()
    n_iter = 0
    while n_iter < cfg.model.epochs and not stop_training:
        net.train(True)
        # for each epoch
        if tb:
            logger.info(f"Epoch {n_iter}")
        c_loss = 0
        for noisy_audio, clean_audio in train_dl:

            clean_audio = clean_audio.float().to(device)
            noisy_audio = noisy_audio.float().to(device)

            optimizer.zero_grad()

            denoised_audio = net(noisy_audio)
            loss = get_loss(time_domain_loss, stft_loss, denoised_audio, clean_audio)

            loss.backward()
            reduced_loss = loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info("Terminated on NaN/INF triggered.")
                break


            c_loss += reduced_loss
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            optimizer.step()

            if tb:
                tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                tb.add_scalar(
                    "Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter
                )
            
    
        if tb:
            logger.info(f"train_loss: {c_loss}")
            tb.add_scalar("Train/Train-Loss", c_loss, n_iter)
        
        # Validation
        net.eval()
        val_c_loss = 0
        with torch.no_grad():
            for noisy_audio, clean_audio in val_dl:
                clean_audio = clean_audio.float().to(device)
                noisy_audio = noisy_audio.float().to(device)
                denoised_audio = net(noisy_audio)
                loss = get_loss(time_domain_loss, stft_loss, denoised_audio, clean_audio)
                val_c_loss += loss.item()
        
        if tb:
            tb.add_scalar("Validation/Validation-Loss", val_c_loss, n_iter)
            logger.info(f"validation_loss: {val_c_loss}")

                # Check for early stopping
        if val_c_loss < best_val_loss:
            best_val_loss = val_c_loss
            best_epoch = n_iter
        elif n_iter - best_epoch >= patience:
            logger.info("Early stopping triggered.")
            stop_training = True

        # save checkpoint
        if cfg.model.log_checkpoints and n_iter > 0 and n_iter % cfg.model.checkpoint_freq == 0:
            checkpoint_name = "{}.pkl".format(n_iter)
            output_dir = pathlib.Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            torch.save(
                {
                    "iter": n_iter,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_time_seconds": int(time.time() - time0),
                },
                output_dir / checkpoint_name,
            )
            logger.info("model at iteration %s is saved" % n_iter)

        n_iter += 1

    # After training, close TensorBoard.
    tb.close()

    return val_c_loss
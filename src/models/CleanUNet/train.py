import pathlib
import logging
import time
import hydra
import omegaconf

import torch
import numpy as np
import keras
import einops

from src.utils import Mode
from src.callbacks import VisualizeCallback
from src.models.CleanUNet.clean_unet_model import CleanUNet
from src.models.CleanUNet.utils import CleanUNetLoss
from src.models.CleanUNet.dataset import CleanUNetDataset, CleanUNetDatasetCSV
from src.models.CleanUNet.validate import visualize_predictions_clean_unet
from src.models.CleanUNet.clean_unet2_model import baseline_model, baseline_unet

from torch.utils.tensorboard import SummaryWriter
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.CleanUNet.stft_loss import MultiResolutionSTFTLoss
from src.metrics import (
    max_amplitude_difference_torch,
    cross_correlation_torch,
    p_wave_onset_difference_torch,
)
from src.utils import LinearWarmupCosineDecay, log_gradient_stats


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
        raise NotImplementedError

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
    else:
        inps = torch.randn((32, 3, 6120))
        tgts = torch.randn((32, 3, 6120))
        train_dataset = torch.utils.data.TensorDataset(inps, tgts)
        val_dataset = train_dataset

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

    train_model(net, optimizer, train_dataset, val_dataset, cfg, tb=tb)

    return 0


def train_model(net, optimizer, train_dataset, val_dataset, cfg, tb=None, sc_lambda=None, mag_lambda=None, stft_lambda=None):

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size
    )

    # loss function
    def get_loss(denoised_audio, clean_audio, stft_lambda, sc_lambda, mag_lambda):

        stft_lambda = stft_lambda if stft_lambda else cfg.model.stft_lambda

        time_domain_loss = torch.nn.L1Loss()
        stft_loss = MultiResolutionSTFTLoss(
            cfg.model.fft_sizes,
            cfg.model.frame_lengths,
            cfg.model.frame_steps,
            sc_lambda= sc_lambda if sc_lambda else cfg.model.sc_lambda,
            mag_lambda=mag_lambda if mag_lambda else cfg.model.mag_lambda,
            transform_stft=False if cfg.model.loss == "stft" else True
        ).to(device)
        if cfg.model.loss == "clean_unet_loss":
            loss = time_domain_loss(denoised_audio, clean_audio)
            spec_loss, mag_loss = stft_loss(denoised_audio, clean_audio)
            loss += ((spec_loss + mag_loss) * cfg.model.stft_lambda)
        elif cfg.model.loss == "stft":
            denoised_audio = einops.rearrange(denoised_audio, "b repeat c t ->(b repeat) t c", repeat=3)
            denoised_audio = torch.clamp(denoised_audio, min=1e-7)
            clean_audio = einops.rearrange(clean_audio, "b repeat c t ->(b repeat) t c", repeat=3)
            spec_loss, mag_loss = stft_loss(denoised_audio, clean_audio)
            loss = (spec_loss + mag_loss) * stft_lambda
            # loss = spec_loss * cfg.model.stft_lambda
        elif cfg.model.loss == "mae":
            loss = time_domain_loss(denoised_audio, clean_audio)
        else:
            raise NotImplementedError

        return loss

    # training
    best_val_loss = float("inf")
    best_epoch = 0
    patience = cfg.model.patience  # Number of epochs to wait for improvement
    stop_training = False

    if cfg.model.lr_schedule:
        scheduler = LinearWarmupCosineDecay(
                        optimizer,
                        lr_max=cfg.model.lr,
                        n_iter=int((20600 // cfg.model.batch_size) * cfg.model.epochs),
                        iteration=0,
                        divider=25,
                        warmup_proportion=0.05,
                        phase=('linear', 'cosine'),
                    )

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
            loss = get_loss(denoised_audio, clean_audio, stft_lambda, sc_lambda, mag_lambda)

            loss.backward()
            reduced_loss = loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info("Terminated on NaN/INF triggered.")
                stop_training=True
                break

            c_loss += reduced_loss
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.model.clipnorm)
            if cfg.model.lr_schedule:
                scheduler.step()
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
        if n_iter % cfg.model.val_freq == 0:
            net.eval()
            with torch.no_grad():
                if cfg.model.snrs:
                    val_losses = []
                    amp_means, amp_stds, cc_means, cc_stds, pw_means, pw_stds = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    for snr in cfg.model.snrs:

                        val_dataset.snr_upper, val_dataset.snr_lower = snr, snr
                        val_dl = torch.utils.data.DataLoader(
                            val_dataset, batch_size=cfg.model.batch_size
                        )
                        val_c_loss = 0
                        max_amp_differences, cc, pw = [], [], []
                        for noisy_audio, clean_audio, shift in val_dl:

                            clean_audio = clean_audio.float().to(device)
                            noisy_audio = noisy_audio.float().to(device)
                            shift = shift.to(device)
                            denoised_audio = net(noisy_audio)
                            # loss
                            loss = get_loss(denoised_audio, clean_audio, stft_lambda, sc_lambda, mag_lambda)
                            val_c_loss += loss.item()
                            # metrics
                            if cfg.model.use_metrics:
                                max_amp_differences.append(
                                    max_amplitude_difference_torch(clean_audio, denoised_audio)
                                )
                                cc.append(cross_correlation_torch(clean_audio, denoised_audio))
                                pw.append(
                                    p_wave_onset_difference_torch(
                                        clean_audio, denoised_audio, shift
                                    )
                                )
                        if cfg.model.use_metrics:
                            max_amp_differences = torch.concatenate(max_amp_differences, dim=0)
                            amp_means.append(max_amp_differences.mean())
                            amp_stds.append(max_amp_differences.std())

                            cc = torch.concatenate(cc, dim=0)
                            cc_means.append(cc.mean())
                            cc_stds.append(cc.std())

                            pw = torch.concatenate(pw, dim=0)
                            pw_means.append(pw.mean())
                            pw_stds.append(pw.std())

                        val_losses.append(val_c_loss)
                else:
                    val_dl = torch.utils.data.DataLoader(
                            val_dataset, batch_size=cfg.model.batch_size
                        )
                    val_c_loss = 0
                    max_amp_differences, cc, pw = [], [], []
                    for noisy_audio, clean_audio, shift in val_dl:
                        clean_audio = clean_audio.float().to(device)
                        noisy_audio = noisy_audio.float().to(device)
                        shift = shift.to(device)
                        denoised_audio = net(noisy_audio)
                        # loss
                        loss = get_loss(denoised_audio, clean_audio, stft_lambda, sc_lambda, mag_lambda)
                        val_c_loss += loss.item()
                        if cfg.model.use_metrics:
                            max_amp_differences.append(
                                max_amplitude_difference_torch(clean_audio, denoised_audio)
                            )
                            cc.append(cross_correlation_torch(clean_audio, denoised_audio))
                            pw.append(
                                p_wave_onset_difference_torch(
                                    clean_audio, denoised_audio, shift
                                )
                            )
                    if cfg.model.use_metrics:
                        max_amp_differences = torch.concatenate(max_amp_differences, dim=0)
                        ma_mean = max_amp_differences.mean()
                        ma_std = max_amp_differences.std()

                        cc = torch.concatenate(cc, dim=0)
                        cc_mean = cc.mean()
                        cc_std = cc.std()

                        pw = torch.concatenate(pw, dim=0)
                        pw_mean = pw.mean()
                        pw_std = pw.std()

                        metrics = {'ma_mean': ma_mean, 'ma_std': ma_std, 'cc_mean': cc_mean, 'cc_std': cc_std, 'pw_mean': pw_mean, 'pw_std': pw_std}
            
            if cfg.model.snrs:
                mean_loss = torch.mean(torch.Tensor(val_losses)).item()
                if tb:
                    for i, snr in enumerate(cfg.model.snrs):
                        tb.add_scalar(
                            f"Validation/Validation-Loss_{snr}", val_losses[i], n_iter
                        )
                        if cfg.model.use_metrics:
                            tb.add_scalar(f"Metrics/Max_Amplitude/Mean_{snr}", amp_means[i], n_iter)
                            tb.add_scalar(f"Metrics/Max_Amplitude/Std_{snr}", amp_stds[i], n_iter)
                            tb.add_scalar(f"Metrics/CC/Mean_{snr}", cc_means[i], n_iter)
                            tb.add_scalar(f"Metrics/CC/Std_{snr}", cc_stds[i], n_iter)
                            tb.add_scalar(f"Metrics/PW/Mean_{snr}", pw_means[i], n_iter)
                            tb.add_scalar(f"Metrics/PW/Std_{snr}", pw_stds[i], n_iter)
                        logger.info(f"validation_loss: {val_losses}")
            else:
                mean_loss = val_c_loss
                if tb:
                    tb.add_scalar(
                            f"Validation/Validation-Loss", val_c_loss, n_iter
                        )
                    logger.info(f"validation_loss: {val_c_loss}")

            # Check for early stopping
            if mean_loss < best_val_loss:
                best_val_loss = mean_loss
                best_epoch = n_iter
            elif n_iter - best_epoch >= patience:
                logger.info(f"Early stopping triggered at epoch: {n_iter}")
                stop_training = True

        # save checkpoint
        if (
            cfg.model.log_checkpoints
            and n_iter > 0
            and n_iter % cfg.model.checkpoint_freq == 0
        ):
            checkpoint_name = "{}.pkl".format(n_iter)
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
    if tb:
        tb.close()

    if cfg.model.save_model:
        torch.save(net.state_dict(), output_dir / "model.pth")

    if cfg.model.use_metrics:
        return val_c_loss, metrics
    else:
        return val_c_loss

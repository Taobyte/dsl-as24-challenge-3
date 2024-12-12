import logging
import pathlib
import time

import einops
import hydra
import numpy as np
import omegaconf
import torch
import accelerate
from torch.utils.tensorboard import SummaryWriter

from src.models.DeepDenoiser.dataset import get_dataloaders_pytorch

from src.metrics import (
    cross_correlation_torch,
    max_amplitude_difference_torch,
    p_wave_onset_difference_torch,
)

from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.CleanUNet.stft_loss import MultiResolutionSTFTLoss
from src.utils import LinearWarmupCosineDecay, log_model_size


logger = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_clean_unet_pytorch(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    train_dl, val_dl = get_dataloaders_pytorch(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accelerator = None
    if cfg.multi_gpu == "accelerate":
        accelerator = accelerate.Accelerator()
        device = accelerator.device

    # Create tensorboard logger.
    tb = None
    if (accelerator is None) or (
        (accelerator is not None) and accelerator.is_main_process
    ):
        tb = SummaryWriter(output_dir)

    # predefine model
    net = CleanUNetPytorch(**cfg.model.architecture).to(device)

    log_model_size(net)

    if cfg.model.checkpoint_model:
        checkpoint = torch.load(cfg.model.checkpoint_model, map_location="cpu")
        net.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully.")

    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.model.lr)

    scheduler = None
    n_obs = cfg.model.subset if cfg.model.subset else 20600
    if cfg.model.lr_schedule:
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            lr_max=cfg.model.lr,
            n_iter=int((n_obs // cfg.model.batch_size) * cfg.model.epochs),
            iteration=0,
            divider=25,
            warmup_proportion=0.05,
            phase=("linear", "cosine"),
        )

    if cfg.multi_gpu == "accelerate":
        if cfg.model.lr_schedule:
            net, optimizer, train_dl, scheduler = accelerator.prepare(
                net, optimizer, train_dl, scheduler
            )
        else:
            net, optimizer, train_dl = accelerator.prepare(net, optimizer, train_dl)

    model = train_model(
        net,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
        cfg,
        tb=tb,
        accelerator=accelerator,
    )

    return model


# loss function
def get_loss(
    denoised_eq: torch.Tensor,
    clean_eq: torch.Tensor,
    stft_lambda: float,
    sc_lambda: float,
    mag_lambda: float,
    cfg: omegaconf.DictConfig,
    accelerator: accelerate.Accelerator = None,
) -> torch.Tensor:
    stft_lambda = stft_lambda if stft_lambda else cfg.model.stft_lambda

    time_domain_loss = torch.nn.L1Loss()
    stft_loss = MultiResolutionSTFTLoss(
        cfg.model.fft_sizes,
        cfg.model.frame_lengths,
        cfg.model.frame_steps,
        sc_lambda=sc_lambda if sc_lambda else cfg.model.sc_lambda,
        mag_lambda=mag_lambda if mag_lambda else cfg.model.mag_lambda,
        transform_stft=False if cfg.model.loss == "stft" else True,
    ).to(device if not cfg.multi_gpu == "accelerate" else accelerator.device)
    if cfg.model.loss == "clean_unet_loss":
        loss = time_domain_loss(denoised_eq, clean_eq)
        spec_loss, mag_loss = stft_loss(denoised_eq, clean_eq)
        loss += (spec_loss + mag_loss) * cfg.model.stft_lambda
    elif cfg.model.loss == "stft":
        denoised_eq = einops.rearrange(
            denoised_eq, "b repeat c t ->(b repeat) t c", repeat=3
        )
        denoised_eq = torch.clamp(denoised_eq, min=1e-7)
        clean_eq = einops.rearrange(clean_eq, "b repeat c t ->(b repeat) t c", repeat=3)
        spec_loss, mag_loss = stft_loss(denoised_eq, clean_eq)
        loss = (spec_loss + mag_loss) * stft_lambda

    elif cfg.model.loss == "mae":
        loss = time_domain_loss(denoised_eq, clean_eq)
    else:
        raise NotImplementedError

    return loss


def train_model(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_dl: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.DataLoader,
    cfg: omegaconf.DictConfig,
    tb: SummaryWriter = None,
    sc_lambda: float = None,
    mag_lambda: float = None,
    stft_lambda: float = None,
    accelerator: accelerate.Accelerator = None,
) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # training
    best_val_loss = float("inf")
    best_epoch = 0
    stop_training = False
    n_iter = 0
    starting_time = time.time()
    while n_iter < cfg.model.epochs and not stop_training:
        time0 = time.time()
        net.train()
        optimizer.zero_grad()
        # for each epoch
        if tb:
            logger.info(f"Epoch {n_iter}")
        c_loss = 0
        for noisy_audio, clean_eq in train_dl:
            clean_eq = clean_eq.float()
            noisy_audio = noisy_audio.float()
            if not cfg.multi_gpu == "accelerate":
                clean_eq = clean_eq.to(device)
                noisy_audio = noisy_audio.to(device)

            denoised_eq = net(noisy_audio)

            loss = get_loss(
                denoised_eq,
                clean_eq,
                stft_lambda,
                sc_lambda,
                mag_lambda,
                cfg,
                accelerator,
            )
            if cfg.multi_gpu == "accelerate":
                accelerator.backward(loss)
            else:
                loss.backward()

            reduced_loss = loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info("Terminated on NaN/INF triggered.")
                stop_training = True
                break

            c_loss += reduced_loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), cfg.model.clipnorm
            )
            if cfg.model.lr_schedule:
                scheduler.step()
            optimizer.step()

            if tb:
                tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                tb.add_scalar(
                    "Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter
                )

        time1 = time.time()
        if tb:
            logger.info(f"train_loss: {c_loss} | time: {time1-time0}")
            tb.add_scalar("Train/Train-Loss", c_loss, n_iter)

        # Validation
        if cfg.model.validation:
            if n_iter > 0 and n_iter % cfg.model.val_freq == 0:
                val_c_loss = validate_model()

        # save checkpoint
        if (
            cfg.model.log_checkpoints
            and n_iter > 0
            and n_iter % cfg.model.checkpoint_freq == 0
        ):
            if accelerator:
                accelerator.wait_for_everyone()
                accelerator.save_model(net, output_dir / f"{n_iter}")
            else:
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

    ending_time = time.time()
    logger.info(f"Total time: {ending_time - starting_time}")

    # After training, close TensorBoard.
    if tb:
        tb.close()

    if cfg.model.save_model:
        if accelerator:
            accelerator.wait_for_everyone()
            accelerator.save_model(net, output_dir / "model.pth")
        else:
            torch.save(net.state_dict(), output_dir / "model.pth")

    return net


def validate_model(net, n_iter, stft_lambda, sc_lambda, mag_lambda, cfg, tb):
    net.eval()
    patience = cfg.model.patience  # Number of epochs to wait for improvement
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
                for noisy_audio, clean_eq, shift in val_dl:
                    clean_eq = clean_eq.float().to(device)
                    noisy_audio = noisy_audio.float().to(device)
                    shift = shift.to(device)
                    denoised_eq = net(noisy_audio)
                    # loss
                    loss = get_loss(
                        denoised_eq,
                        clean_eq,
                        stft_lambda,
                        sc_lambda,
                        mag_lambda,
                        cfg,
                    )
                    val_c_loss += loss.item()
                    # metrics
                    if cfg.model.use_metrics:
                        max_amp_differences.append(
                            max_amplitude_difference_torch(clean_eq, denoised_eq)
                        )
                        cc.append(cross_correlation_torch(clean_eq, denoised_eq))
                        pw.append(
                            p_wave_onset_difference_torch(clean_eq, denoised_eq, shift)
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
            for noisy_audio, clean_eq, shift in val_dl:
                clean_eq = clean_eq.float().to(device)
                noisy_audio = noisy_audio.float().to(device)
                shift = shift.to(device)
                denoised_eq = net(noisy_audio)
                # loss
                loss = get_loss(
                    denoised_eq,
                    clean_eq,
                    stft_lambda,
                    sc_lambda,
                    mag_lambda,
                )
                val_c_loss += loss.item()

                if cfg.model.use_metrics:
                    max_amp_differences.append(
                        max_amplitude_difference_torch(clean_eq, denoised_eq)
                    )
                    cc.append(cross_correlation_torch(clean_eq, denoised_eq))
                    pw.append(
                        p_wave_onset_difference_torch(clean_eq, denoised_eq, shift)
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

                metrics = {
                    "ma_mean": ma_mean,
                    "ma_std": ma_std,
                    "cc_mean": cc_mean,
                    "cc_std": cc_std,
                    "pw_mean": pw_mean,
                    "pw_std": pw_std,
                }

    if cfg.model.snrs:
        mean_loss = torch.mean(torch.Tensor(val_losses)).item()
        if tb:
            for i, snr in enumerate(cfg.model.snrs):
                tb.add_scalar(
                    f"Validation/Validation-Loss_{snr}", val_losses[i], n_iter
                )
                if cfg.model.use_metrics:
                    tb.add_scalar(
                        f"Metrics/Max_Amplitude/Mean_{snr}",
                        amp_means[i],
                        n_iter,
                    )
                    tb.add_scalar(
                        f"Metrics/Max_Amplitude/Std_{snr}", amp_stds[i], n_iter
                    )
                    tb.add_scalar(f"Metrics/CC/Mean_{snr}", cc_means[i], n_iter)
                    tb.add_scalar(f"Metrics/CC/Std_{snr}", cc_stds[i], n_iter)
                    tb.add_scalar(f"Metrics/PW/Mean_{snr}", pw_means[i], n_iter)
                    tb.add_scalar(f"Metrics/PW/Std_{snr}", pw_stds[i], n_iter)
                logger.info(f"validation_loss: {val_losses}")
    else:
        mean_loss = val_c_loss
        if tb:
            tb.add_scalar("Validation/Validation-Loss", val_c_loss, n_iter)
            if cfg.model.use_metrics:
                tb.add_scalar("Metrics/Max_Amplitude/Mean", ma_mean, n_iter)
                tb.add_scalar("Metrics/Max_Amplitude/Std", ma_std, n_iter)
                tb.add_scalar("Metrics/CC/Mean", cc_mean, n_iter)
                tb.add_scalar("Metrics/CC/Std", cc_std, n_iter)
                tb.add_scalar("Metrics/PW/Mean", pw_mean, n_iter)
                tb.add_scalar("Metrics/PW/Std", pw_std, n_iter)
            logger.info(f"validation_loss: {val_c_loss}")

    # Check for early stopping
    if mean_loss < best_val_loss:
        best_val_loss = mean_loss
        best_epoch = n_iter
    elif patience and n_iter - best_epoch >= patience:
        logger.info(f"Early stopping triggered at epoch: {n_iter}")
        stop_training = True

    return val_c_loss

import logging
import pathlib
import time

import hydra
import omegaconf
import torch
from torch.utils.tensorboard import SummaryWriter
import einops

from src.utils import get_trained_model, Model
from src.models.DeepDenoiser.dataset import get_dataloaders_pytorch
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.stft import MultiResolutionSTFTLoss
from src.utils import LinearWarmupCosineDecay, log_model_size, EarlyStopper


logger = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_clean_unet_pytorch(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    train_dl, val_dl = get_dataloaders_pytorch(cfg, model=Model.CleanUNet)
    tb = SummaryWriter(output_dir)

    if cfg.model.load_checkpoint:
        net = get_trained_model(cfg, Model.CleanUNet)
    else:
        net = CleanUNetPytorch(**cfg.model.architecture).to(device)

    log_model_size(net)

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

        logger.info(
            f"n_iter: {int((n_obs // cfg.model.batch_size) * cfg.model.epochs)}"
        )

    model = train_model(
        net,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
        cfg,
        tb=tb,
    )

    return model


def get_loss(
    denoised_eq: torch.Tensor,
    clean_eq: torch.Tensor,
    stft_lambda: float,
    sc_lambda: float,
    mag_lambda: float,
    cfg: omegaconf.DictConfig,
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
    ).to(device)
    if cfg.model.loss == "clean_unet_loss":
        loss = time_domain_loss(denoised_eq, clean_eq)
        spetrain_loss, mag_loss = stft_loss(denoised_eq, clean_eq)
        loss += (spetrain_loss + mag_loss) * cfg.model.stft_lambda
    elif cfg.model.loss == "stft":
        denoised_eq = einops.rearrange(
            denoised_eq, "b repeat c t ->(b repeat) t c", repeat=3
        )
        denoised_eq = torch.clamp(denoised_eq, min=1e-7)
        clean_eq = einops.rearrange(clean_eq, "b repeat c t ->(b repeat) t c", repeat=3)
        spetrain_loss, mag_loss = stft_loss(denoised_eq, clean_eq)
        loss = (spetrain_loss + mag_loss) * stft_lambda

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
    val_dl: torch.utils.data.DataLoader,
    cfg: omegaconf.DictConfig,
    tb: SummaryWriter = None,
    sc_lambda: float = None,
    mag_lambda: float = None,
    stft_lambda: float = None,
) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if cfg.model.patience:
        early_stopper = EarlyStopper(cfg.model.patience, cfg.model.min_delta)

    n_iter = 0
    starting_time = time.time()
    for n_iter in range(cfg.model.epochs):
        time0 = time.time()
        net.train()
        optimizer.zero_grad()
        if tb:
            logger.info(f"Epoch {n_iter}")
        train_loss = 0
        for eq, noise in train_dl:
            eq = eq.float().to(device)
            noise = noise.float().to(device)
            noisy_eq = eq + noise
            denoised_eq = net(noisy_eq)

            loss = get_loss(denoised_eq, eq, stft_lambda, sc_lambda, mag_lambda, cfg)
            loss.backward()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info("Terminated on NaN/INF triggered.")
                break

            train_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), cfg.model.clipnorm
            )
            optimizer.step()
            if scheduler:
                scheduler.step()

            tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
            tb.add_scalar(
                "Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter
            )

        time1 = time.time()
        logger.info(f"train_loss: {train_loss} | time: {time1-time0}")
        tb.add_scalar("Train/Train-Loss", train_loss, n_iter)

        if n_iter > 0 and n_iter % cfg.model.val_freq == 0:
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for eq, noise in val_dl:
                    eq = eq.float().to(device)
                    noise = noise.float().to(device)
                    noisy_eq = eq + noise
                    denoised_eq = net(noisy_eq)
                    loss = get_loss(
                        denoised_eq,
                        eq,
                        stft_lambda,
                        sc_lambda,
                        mag_lambda,
                        cfg,
                    )
                    val_loss += loss.item()
                logger.info(f"val_loss: {val_loss}")
                tb.add_scalar("Validation/Val-Loss", val_loss, n_iter)

                if early_stopper.early_stop(val_loss):
                    logger.info(f"Early stopping triggered at epoch {n_iter}.")
                    break
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

    ending_time = time.time()
    logger.info(f"Total time: {ending_time - starting_time}")

    tb.close()

    torch.save(net.state_dict(), output_dir / "model.pkl")

    return net

import os
import pathlib
import logging
import time

import torch
import hydra
import omegaconf
from tqdm import tqdm

from src.utils import log_model_size
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_dataloaders_pytorch
from models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser

logger = logging.getLogger()


def fit_deep_denoiser_pytorch(cfg: omegaconf.DictConfig) -> torch.nn.Module:
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    tb = SummaryWriter(output_dir)

    model = DeepDenoiser(**cfg.model.architecture).to(device)
    log_model_size(model)

    optimizer = torch.optim.AdamW(model.parameters(), cfg.model.lr)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    train_dl, val_dl = get_dataloaders_pytorch(cfg, subset=cfg.model.subset)
    if cfg.model.subset:
        logger.info(f"using {cfg.model.subset} data points for training")

    logger.info("Start training DeepDenoiser")
    best_val_loss = float("inf")
    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        train_loss = 0.0
        time0 = time.time()
        for iter, (eq, noise, mask) in enumerate(train_dl):
            optimizer.zero_grad()
            eq, noise, mask = eq.float(), noise.float(), mask.float()
            noisy_eq = (eq + noise).to(device)
            mask = mask.to(device)
            predicted_mask = model(noisy_eq)
            loss = bce_loss(predicted_mask, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.model.clip_norm
            )
            tb.add_scalar("gradient_norm", grad_norm, epoch)
            tb.add_scalar("step", loss.item(), epoch)
            logger.info(f"Loss at step {iter}: {loss.item()}")
        time1 = time.time()
        train_loss /= len(train_dl)
        tb.add_scalar("train_loss", train_loss, epoch)
        logger.info(
            f"Epoch {epoch} |  train_loss: {train_loss} | grad_norm: {grad_norm} | epoch_time: {time1 - time0}"
        )

        # validation
        if epoch % cfg.model.val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for eq, noise, mask in tqdm(val_dl, total=len(val_dl)):
                    eq, noise, mask = eq.float(), noise.float(), mask.float()
                    noisy_eq = (eq + noise).to(device)
                    mask = mask.to(device)
                    predicted_mask = model(noisy_eq)
                    loss = bce_loss(predicted_mask, mask)
                    val_loss += loss.item()
                val_loss /= len(val_dl)
                tb.add_scalar("validation_loss", val_loss, epoch)

                logger.info(f"Epoch {epoch} |  val_loss: {val_loss}")

                if val_loss < best_val_loss:
                    torch.save(
                        model.state_dict(),
                        output_dir / f"epoch_{epoch}.pth",
                    )
                    best_val_loss = val_loss
                    logger.info(
                        f"New model saved at epoch: {epoch} | best_val_loss: {best_val_loss}"
                    )

    torch.save(
        model.state_dict(), output_dir / f"checkpoints/deep_denoiser_{epoch}.pth"
    )

    return model

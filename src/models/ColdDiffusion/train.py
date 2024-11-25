import os
import pathlib
import hydra
import omegaconf 

import torch
import numpy as np
import keras
import wandb

from utils import Mode
from models.ColdDiffusion.ColdDiffusion_keras import ColdDiffusion
from models.ColdDiffusion.dataset import ColdDiffusionDataset

def fit_cold_diffusion(cfg: omegaconf.DictConfig) -> keras.Model:

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    if cfg.user.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="earthquake denoising",
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.model.lr,
                "architecture": "ColdDiffusion",
                "dataset": "SED dataset",
                "epochs": cfg.model.epochs,
                "batch_size": cfg.model.batch_size,
                "dim": cfg.model.dim,
                "dim_multiples": cfg.model.dim_multiples,
                "num_workers": cfg.model.num_workers,
            },
        )
        wandb.run.name = "{}".format(os.getcwd().split('outputs/')[-1])

    model = ColdDiffusion(
        dim=int(cfg.model.dim), 
        dim_multiples=list(cfg.model.dim_multiples),
        in_dim=3,
        out_dim=3,
        attn_dim_head=int(cfg.model.attn_dim_head),
        attn_heads=int(cfg.model.attn_heads),
        resnet_norm_groups=int(cfg.model.resnet_norm_groups),
    )
    if cfg.model.continue_from_pretrained:
        model = keras.saving.load_model(cfg.model.pretrained_path, custom_objects={"ColdDiffusion": model})

    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.model.lr),
        # metrics=[keras.metrics.MeanSquaredError(name="Part1"),
        #          ],
        # run_eagerly=True,
    )
    # build model
    # time = keras.random.randint((cfg.model.batch_size,), 0, cfg.model.T)
    x = keras.random.normal((cfg.model.batch_size, 3, cfg.model.signal_length))
    model(x)#, None)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "checkpoints/model_at_epoch_{epoch}.keras"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]
    if cfg.user.wandb:
        wandb_callbacks = [wandb.integration.keras.WandbMetricsLogger(log_freq="batch")]
        callbacks = callbacks.extend(wandb_callbacks)
    
    train_dataset = ColdDiffusionDataset(cfg.user.data.train_file, shape=(20230, 6, cfg.model.signal_length))
    val_dataset = ColdDiffusionDataset(cfg.user.data.val_file, shape=(4681, 6, cfg.model.signal_length))

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size, num_workers=cfg.model.num_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.model.batch_size, num_workers=cfg.model.num_workers,
    )

    model.summary() 
    model.fit(
        train_dl, 
        epochs=cfg.model.epochs, 
        validation_data=val_dl, 
        callbacks=callbacks,
        batch_size=cfg.model.batch_size,
    )

    return model


def custom_metric(loss):
    return loss


class ColdDiffusionLoss(keras.losses.Loss):
    def __init__(self, penalty: float=0.5, loss_type: str = "MSE"):
        super().__init__()
        self.penalty = penalty
        self.loss_type = loss_type
        if loss_type == "MSE":
            self.fct = keras.losses.MeanSquaredError()
        else:
            raise Exception("CustomLossException --> not a valid loss_type")

    def call(self, eq, denoised_eqs):
        print(denoised_eqs.shape)
        assert denoised_eqs.shape[1] == 6, "Custom Assertion: Dimensions do not match in loss"
        l1 = self.fct(eq, denoised_eqs[:,:3,:])
        l2 = self.fct(eq, denoised_eqs[:,3:,:])

        return l1 + self.penalty*l2

    def get_config(self):
        config = super().get_config()
        config.update({
                "penalty": self.penalty,
                "loss_type": self.loss_type,
        })
        return config

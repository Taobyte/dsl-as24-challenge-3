import random
import logging
import pathlib

import jax
import hydra
import omegaconf
import numpy as np
import tensorflow as tf
import torch as th

from train import train_model
from tuner import tune_model, tune_model_optuna
from validate import compute_metrics
from plot import compare_two, visualize_predictions


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):

    print(omegaconf.OmegaConf.to_yaml(cfg))
    log.info(omegaconf.OmegaConf.to_yaml(cfg))

    # Set see (default: 123)
    seed = cfg.model.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    th.manual_seed(seed)
    # jax_key = jax.random.PRNGKey(seed)

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if cfg.tuner:

        if cfg.use_optuna:
            best_params = tune_model_optuna(cfg)
        else:
            best_hypers = tune_model(cfg)

    else:

        if cfg.training:
            model = train_model(cfg)
            model.save(
                output_dir
                / f"{cfg.model.model_name}_lr_{cfg.model.lr}_e{cfg.model.epochs}.keras"
            )
        elif cfg.test:
            model_df, butterworth_df = compute_metrics(cfg)
            if cfg.plot.metrics:
                compare_two(
                output_dir / "metrics_model.csv",
                output_dir / "metrics_butterworth.csv",
                label1=cfg.model.model_name,
                label2="Butterworth",
            )
        elif cfg.plot.metrics:
            compare_two(
                cfg.user.metrics_model_path,
                cfg.user.metrics_butterworth_path,
                label1=cfg.model.model_name,
                label2="Butterworth",
            )
        elif cfg.plot.visualization:
            visualize_predictions(cfg)


if __name__ == "__main__":
    main()

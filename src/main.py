import random
import numpy as np
import tensorflow as tf
import torch as th
import jax
import os
from train_validate import train_model, compute_metrics
from tuner import tune_model

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

"""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "jax"
"""

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # Set see (default: 123)
    seed = cfg.model.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    th.manual_seed(seed)
    jax_key = jax.random.PRNGKey(seed)

    if cfg.tuner:

        best_hypers = tune_model(cfg)

    else:

        if cfg.training:
            model = train_model(cfg)
        else:
            loss = compute_metrics(cfg)

if __name__ == '__main__':
    main()

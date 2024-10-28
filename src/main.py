import random
import logging

import jax
import hydra
import omegaconf
import numpy as np
import tensorflow as tf
import torch as th

from train import train_model
from tuner import tune_model
from validate import compute_metrics
from plot import compare_two


log = logging.getLogger(__name__)

"""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "jax"
"""

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):

    print(omegaconf.OmegaConf.to_yaml(cfg))

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
            df1, df2 = compute_metrics(cfg)
            if cfg.plot:
                compare_two(df1, df2, label1=cfg.model.model_name, label2="Butterworth")               


if __name__ == '__main__':
    main()

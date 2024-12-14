import random
import logging
import pathlib

import hydra
import omegaconf
import numpy as np
import torch as th

from train import train_model
from validate import compute_metrics, create_prediction_csv
from plot import visualize_predictions, overlay_plot, metrics_plot


def setup_logging(cfg: omegaconf.DictConfig):
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    log_file = output_dir / "main.log"

    logging.basicConfig(
        filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s"
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    setup_logging(cfg)
    # logging.info(omegaconf.OmegaConf.to_yaml(cfg))

    # Set see (default: 123)
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if cfg.training:
        model = train_model(cfg)

    elif cfg.test:
        """
        model_df, butterworth_df = compute_metrics(cfg)
        if cfg.plot.metrics:
            compare_model_and_baselines(
                output_dir / "metrics_model.csv",
                output_dir / "metrics_butterworth.csv",
                cfg.user.metrics_deepdenoiser_path,
                label1=cfg.model.model_name,
                label2="Butterworth",
                label3="DeepDenoiser",
            )
        """
        # df = create_prediction_csv(cfg)
        compute_metrics(cfg)

    elif cfg.plot.metrics:
        metrics_plot(cfg)
    elif cfg.plot.visualization:
        visualize_predictions(cfg)
    elif cfg.plot.overlay_plot:
        overlay_plot(cfg)


if __name__ == "__main__":
    main()

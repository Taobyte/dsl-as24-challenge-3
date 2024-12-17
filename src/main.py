import random
import logging
import pathlib

import hydra
import omegaconf
import numpy as np
import torch as th

from train import train_model
from validate import compute_metrics, create_prediction_csv
from plot import visualize_predictions, overlay_plot, metrics_plot, plot_training_run
from dayplots import plot_day
from src.utils import wilcoxon_test, test_inference_speed


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

    if cfg.training:
        train_model(cfg)
    elif cfg.test:
        compute_metrics(cfg)
    elif cfg.predictions:
        create_prediction_csv(cfg)
    elif cfg.inference_speed.test:
        test_inference_speed(cfg)
    elif cfg.wilcoxon_test:
        wilcoxon_test(cfg)
    elif cfg.plot.metrics:
        metrics_plot(cfg)
    elif cfg.plot.visualization:
        visualize_predictions(cfg)
    elif cfg.plot.overlay_plot.plot:
        overlay_plot(cfg)
    elif cfg.plot.training_run.plot:
        plot_training_run(cfg)
    elif cfg.plot.dayplot:
        plot_day(cfg)


if __name__ == "__main__":
    main()

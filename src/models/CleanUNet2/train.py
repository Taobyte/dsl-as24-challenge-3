import pathlib
import logging
import time
import hydra
import omegaconf

import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils import Mode

from src.models.CleanUNet2.clean_specnet import CleanSpecNet
from src.models.CleanUNet.dataset import CleanUNetDatasetCSV
from src.models.CleanUNet.stft_loss import MultiResolutionSTFTLoss
from src.models.CleanUNet.train import train_model

logger = logging.getLogger() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_clean_specnet(cfg: omegaconf.DictConfig):

    if cfg.model.clean_specnet:
        assert cfg.model.loss == "stft"
        assert not cfg.model.use_metrics
    
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
            spectogram=True,
            fft_size=cfg.model.fft_sizes[0],
            hop_size=cfg.model.frame_steps[0],
            win_length=cfg.model.frame_lengths[0]
        )
        val_dataset = CleanUNetDatasetCSV(
            cfg.user.data.csv_path,
            cfg.model.signal_length,
            cfg.model.snr_lower,
            cfg.model.snr_upper,
            Mode.TEST,
            data_format="channel_first",
            spectogram=True,
            fft_size=cfg.model.fft_sizes[0],
            hop_size=cfg.model.frame_steps[0],
            win_length=cfg.model.frame_lengths[0]
        )
    else: 
        inps = torch.randn((32, 3, 6120))
        tgts = torch.randn((32, 3, 64, 256))
        train_dataset = torch.utils.data.TensorDataset(inps, tgts)
        val_dataset = train_dataset
    
    # predefine model
    net = CleanSpecNet(
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


def fit_clean_unet2(cfg: omegaconf.DictConfig):
    raise NotImplementedError


import pytest
import torch 

from hydra import initialize, compose
from src.utils import Mode
from src.models.CleanUNet.dataset import CleanUNetDatasetCSV

@pytest.fixture(scope="module")
def cfg():
    # Initialize Hydra with the config path relative to the project root
    with initialize(config_path="../../src/conf"):
        # Compose the configuration using the config file
        cfg = compose(config_name="config")
        return cfg


def test_clean_specnet_dataset(cfg):
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

    dl = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    noisy, eq, shift = next(iter(dl))

    assert noisy.shape == (1, 3, 6120)
    assert eq.shape == (1, 3, 64, 256)
    assert isinstance(shift.item(), int)
from torch.utils.data import DataLoader
from src.models.DeepDenoiser.dataset import (
    InputSignals,
    EventMasks,
    get_signal_noise_assoc,
)
from src.utils import Mode
from src.models.CleanUNet.dataset import CleanUNetDataset


def test_input_assoc_list(params):
    assoc = get_signal_noise_assoc(
        params["signal_path"], params["noise_path"], Mode.VALIDATION
    )

    assert (
        len(assoc[0]) == 4
        and isinstance(assoc[0][0], str)
        and isinstance(assoc[0][1], str)
        and isinstance(assoc[0][2], float)
        and isinstance(assoc[0][3], int)
    )


def test_input_signals(params):
    assoc = get_signal_noise_assoc(
        params["signal_path"], params["noise_path"], Mode.VALIDATION
    )
    input_signals = InputSignals(assoc)

    assert len(input_signals[0]) == 3


def test_output_event_masks(params):
    assoc = get_signal_noise_assoc(
        params["signal_path"], params["noise_path"], Mode.VALIDATION
    )

    event_masks = EventMasks(assoc)

    assert event_masks[0].shape == (6, 256, 64)


def test_clean_unet_dataset():
    dataset = CleanUNetDataset(
        "/cluster/scratch/ckeusch/data/signal/" + "train",
        "/cluster/scratch/ckeusch/data/noise/" + "train",
        6120,
    )

    assert True


import pytest

signal_folder = (
    "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/"
)
noise_folder = (
    "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/"
)


@pytest.fixture(scope="session")
def params():
    return {
        'signal_path': signal_folder,
        'noise_path': noise_folder,
    }

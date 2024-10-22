import pytest

SNR_LIST = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.6,.0.7,0.8,0.9,1.0] 

@pytest.fixture(scope="session")
def global_params():
    return {
        "snr_list": SNR_LIST,
    }
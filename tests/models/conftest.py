import pytest

SNR_LIST = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.7,0.8,0.9,1.0] 
signal_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal"
noise_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise"
result_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/predictions/DeepDenoiser/"
model_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/models/DeepDenoiser/model_at_epoch_3.keras"
model_name = "deep_denoiser_res_test.pkl"

@pytest.fixture(scope="session")
def global_params():
    return {
        "snr_list": SNR_LIST,
        'signal_path': signal_path,
        'noise_path': noise_path,
        'result_path': result_path,
        'model_path': model_path, 
        'model_name': model_name
    }
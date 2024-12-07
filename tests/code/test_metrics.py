import numpy as np
import torch

from src.metrics import CCMetric, AmpMetric
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.metrics import max_amplitude_difference_torch, cross_correlation_torch, p_wave_onset_difference_torch


def test_cc_metric():
    
    metric = CCMetric()
    y_true = np.zeros((32,3,6120))
    y_pred = np.zeros((32,3,6120))
    metric.update_state(y_true, y_pred)
    cc = metric.result()
    assert cc >= -1.0 and cc <= 1.0

def test_amp_metric():
    
    metric = AmpMetric()
    y_true = np.zeros((32,3,6120))
    y_pred = np.zeros((32,3,6120))
    metric.update_state(y_true, y_pred)
    amp_ratio = metric.result()
    assert amp_ratio >= 0.0

def test_amp_metric_func_torch():

    net = CleanUNetPytorch(channels_input=3, channels_output=3,channels_H = 4, tsfm_n_layers=1)
    
    input = torch.randn((4,3,6120))
    gt = torch.randn((4,3,6120))
    denoised = net(input)

    max_amp_result = max_amplitude_difference_torch(gt, denoised)

    assert max_amp_result.shape == (4,)


def test_cc_func_torch_shape():

    net = CleanUNetPytorch(channels_input=3, channels_output=3,channels_H = 4, tsfm_n_layers=1)
    
    input = torch.randn((4,3,6120))
    gt = torch.randn((4,3,6120))
    denoised = net(input)
    
    cc_result = cross_correlation_torch(gt, denoised)

    assert cc_result.shape == (4,)

def test_cc_func_torch():
    B = 4
    input = torch.randn((B,3,6120))  
    m, _ = torch.max(input, dim=-1)
    input /= (m.unsqueeze(-1) + 1e-12) 
    
    cc_result = cross_correlation_torch(input, input)

    print(cc_result)

    assert torch.all((cc_result == torch.Tensor([1.0 for _ in range(B)])))

def test_p_wave_func_torch():
    B = 4
    input = torch.randn((B,3,6120))  
    gt = input

    result = p_wave_onset_difference_torch(input, gt, 300)

    assert result.shape == (B,) 

    



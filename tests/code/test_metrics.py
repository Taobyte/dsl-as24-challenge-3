import numpy as np

from src.metrics import CCMetric, AmpMetric

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


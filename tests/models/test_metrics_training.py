import numpy as np
import keras

from src.models.DeepDenoiser.deep_denoiser_model_2 import UNet
from src.metrics import CCMetric

def test_cc_metric_training():

    model = UNet()

    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[CCMetric()])

    data = np.random.random((1, 3, 6120))
    labels = np.random.random((1, 6, 256, 64))

    model.fit(data, labels, epochs=3)

    assert True
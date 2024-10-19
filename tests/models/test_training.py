import numpy as np
import keras
from src.models.DeepDenoiser.deep_denoiser_model import Unet2D
from src.data import get_dataloaders


def test_shapes():

    assert False


def test_decreasing_loss():
    pass


def test_overfit_on_batch():

    learning_rate = 0.001

    model = Unet2D()
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            keras.metrics.BinaryCrossentropy(
                name="binary_crossentropy", dtype=None, from_logits=True
            )
        ],
    )

    signal_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal/train"
    )
    noise_folder = (
        "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise/train"
    )

    train_dl, val_dl = get_dataloaders(signal_folder, noise_folder, 32, 32)

    epochs = 50
    model.fit(train_dl, epochs=epochs, validation_data=val_dl)
    loss = model.evaluate(train_dl, batch_size=32, verbose=2, steps=1)

    assert loss < 0.0001


def test_different_devices():
    pass

import keras
import torch as th
from torch.utils.data import DataLoader
import numpy as np 
import os 
# from src.models.MarsQ.marsq_model import MarsQNetwork
from data import get_dataloaders

os.environ["KERAS_BACKEND"] = "jax"


def train_model(signal_path: str, noise_path: str) -> keras.Model:

    model = keras.Sequential(
        [
        keras.layers.Input(shape=(3,6000)),
        keras.layers.Dense(6000),
        ]
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.MeanSquaredError(),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    train_dl, validation_dl = get_dataloaders(signal_path, noise_path, batch_size=32)
    
    epochs = 100 

    model.fit(train_dl, epochs=epochs, validation_data=validation_dl, callbacks=callbacks)

    model.evaluate(validation_dl, batch_size=32, verbose=2, steps = 1)

    return model


if __name__ == '__main__':

    signal_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal"
    noise_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise"

    model = train_model(signal_path, noise_path)
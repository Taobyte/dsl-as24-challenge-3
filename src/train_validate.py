import keras
import torch as th
from torch.utils.data import DataLoader
import numpy as np 
import os 
from models.DeepDenoiser.deep_denoiser_model import Unet2D
from data import get_dataloaders

import matplotlib.pyplot as plt


def train_model(args) -> keras.Model:

    np.random.seed(args.seed)

    model = Unet2D()
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        metrics=[
            keras.metrics.BinaryCrossentropy(
                name="binary_crossentropy", dtype=None, from_logits=True
            )

        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=args.path_model + "/model_at_epoch_{epoch}.keras"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    train_dl, validation_dl = get_dataloaders(args.signal_path, args.noise_path, args.batch_size, args.length_dataset)
    
    model.fit(train_dl, epochs=args.epochs, validation_data=validation_dl, callbacks=callbacks)

    model.evaluate(validation_dl, batch_size=32, verbose=2, steps = 1)

    return model


def test_model(args):
    
    _, validation_dl = get_dataloaders(args.signal_path, args.noise_path, args.batch_size, args.length_dataset)
    
    model = keras.models.load_model(args.path_model)
    x, y = next(iter(validation_dl))
    predictions = model.predict(x)

    print(predictions)
    print(y)

    fig, axs = plt.subplots(1, 2, figsize=(30, 30))

    cax0 = axs[0].imshow(predictions[0, :, :, 0], cmap='viridis', interpolation='none')  # 'viridis' is a colormap
    cax1 = axs[1].imshow(y[0, :, :, 0], cmap='plasma', interpolation='none')
    axs[0].set_title("Prediction 2D heatmap")
    axs[1].set_title("Ground Truth 2D heatmap")

    fig.colorbar(cax0, ax=axs[0])
    fig.colorbar(cax1, ax=axs[1])

    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    plt.show()

    return predictions


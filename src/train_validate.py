import keras
import torch as th
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
import numpy as np 
import os 
from models.DeepDenoiser.deep_denoiser_model import Unet2D
from data import get_dataloaders

import matplotlib.pyplot as plt
import scipy


def train_model(args) -> keras.Model:

    # set seed (default: 123)
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
    x, ground_truth = next(iter(validation_dl))
    predictions = sigmoid(th.from_numpy(model.predict(x))).numpy()

    print(predictions)
    print(ground_truth)

    def inverse_stft(signal):

        t, time_domain_signal = scipy.signal.istft(
            signal,
            fs=100,
            nperseg=30,
            nfft=60,
            boundary='zeros',
        )

        return time_domain_signal


    time_domain_pred = inverse_stft(predictions[0, :, :, 0])
    time_domain_ground_truth = inverse_stft(ground_truth[0, :, :, 0])
    time = list(range(time_domain_pred.shape[0]))

    fig, axs = plt.subplots(2, 2, figsize=(30, 30))

    # Plot time domain signal
    axs[0][0].plot(time, time_domain_pred)
    axs[1][0].plot(time, time_domain_ground_truth)

    # Plot masks
    cax0 = axs[0][1].imshow(predictions[0, :, :, 0], cmap='plasma', interpolation='none')  # 'viridis' is a colormap
    cax1 = axs[1][1].imshow(ground_truth[0, :, :, 0], cmap='plasma', interpolation='none')
    
    axs[0][0].set_title("Prediction time domain")
    axs[1][0].set_title("Ground Truth time domain")
    
    axs[0][1].set_title("Prediction stft mask")
    axs[1][1].set_title("Ground Truth stft mask")

    fig.colorbar(cax0, ax=axs[0][1])
    fig.colorbar(cax1, ax=axs[1][1])

    axs[0][1].invert_yaxis()
    axs[1][1].invert_yaxis()

    plt.show()

    return predictions


import keras
import torch as th
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
import numpy as np 
from numpy import ndarray
import os 
from argparse import Namespace
from models.DeepDenoiser.deep_denoiser_model import Unet2D
from data import SeismicDataset, get_dataloaders

import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
import optuna


def train_model(args: Namespace) -> keras.Model:

    # set seed (default: 123)
    np.random.seed(args.seed)

    if args.deepdenoiser:

        model = fit_deep_denoiser(args)

        return model
    
    elif args.butterworth:

        best_params = get_best_params(args)
    

    return None


def test_model(args: Namespace) -> ndarray:

    if args.deepdenoiser:
    
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
    
    return np.zeros(1)


def fit_deep_denoiser(args: Namespace) -> keras.Model:
    
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


def get_best_params(args: Namespace) -> dict:

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # Apply the filter
    def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    def objective(trial):

        lowcut = trial.suggest_float('low_cutoff', 1, 15)
        highcut = trial.suggest_float('high_cutoff', 20, 50)
        order = trial.suggest_int('order', 2, 6)

        fs = 100

        dataset = SeismicDataset(args.signal_path, args.noise_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset_length = len(dataset)
        
        mean = 0
        for i, batch in enumerate(dataloader):
            
            noisy_eq, ground_truth = batch
            noisy_eq_np = noisy_eq.numpy()

            filtered_signals = []
            for i in range(len(noisy_eq)):
                filtered_signal = apply_bandpass_filter(noisy_eq_np[i], lowcut, highcut, fs, order)
                filtered_signals.append(filtered_signal)

            # Convert the filtered signals back to a PyTorch tensor
            filtered_eq = th.tensor(np.array(filtered_signals))

            mean += (1/dataset_length) * th.sum(th.square(ground_truth - filtered_eq))
        
        return mean
    

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    return study
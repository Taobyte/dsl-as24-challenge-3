import keras
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
import scipy.optimize

from argparse import Namespace

import wandb
from wandb.integration.keras import WandbMetricsLogger

from models.DeepDenoiser.deep_denoiser_model import Unet2D
from data import get_dataloaders, get_signal_noise_assoc, EventMasks, InputSignals, CombinedDeepDenoiserDataset
from utils import Mode


def train_model(args: Namespace) -> keras.Model:

    if args.butterworth:
        
        params = fit_butterworth(args)
        print(params)
        return params

    elif args.deepdenoiser:

        model = fit_deep_denoiser(args)
        return model

    
    return None


def test_model(args: Namespace) -> pd.DataFrame:

    signal_path = ""
    noise_path = ""
    result_path = ""
    model_path = ""
    model_name = "DeepDenoiser"
    batch_size = 32

    snrs = [0.1 * i for i in range(1,11)]
    signal_length = 6120

    assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.TEST)

    eq_traces = []
    noise_traces = []
    shifts = []
    
    for eq_path, noise_path, _, shift in assoc:

        eq = np.load(eq_path, allow_pickle=True)
        noise = np.load(noise_path, allow_pickle=True)
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + signal_length]
        N_eq = eq["earthquake_waveform_N"][event_shift : event_shift + signal_length]
        E_eq = eq["earthquake_waveform_E"][event_shift : event_shift + signal_length]
        eq_stacked = np.stack([Z_eq, N_eq, E_eq], axis=0)

        Z_noise = noise["noise_waveform_Z"][:signal_length]
        N_noise = noise["noise_waveform_N"][:signal_length]
        E_noise = noise["noise_waveform_E"][:signal_length]
        noise_stacked = np.stack([Z_noise, N_noise, E_noise], axis=0)

        eq_traces.append(eq_stacked)
        noise_traces.append(noise_stacked)
        shifts.append(shift)
    
    model = keras.saving.load_model(model_path)
    predictions = {}
    for snr in snrs:
        test_dataset = InputSignals(assoc, Mode.TEST, snr)
        test_dl = DataLoader(test_dataset, batch_size, shuffle=False)
        predicted_mask = model.predict(test_dl)

















    if args.deepdenoiser:

        val_assoc = get_signal_noise_assoc(args.signal_path, args.noise_path, train=False)
        val_dl = DataLoader(CombinedDeepDenoiserDataset(InputSignals(val_assoc), EventMasks(val_assoc)), batch_size=1, shuffle=False)
    
        model = keras.models.load_model(args.path_model)
        x, ground_truth = next(iter(val_dl))
        predictions = model.predict(x)

        print(x.shape)
        print(ground_truth.shape)

        real, imag = keras.ops.stft(x, 100, 24, 126)
        stft = np.concatenate([real,imag], axis=1)

        masked_out = stft * predictions

        time_domain_pred = keras.ops.istft((masked_out[0][0],masked_out[0][3]), 100, 24, 126)

        time = list(range(time_domain_pred.shape[0]))

        fig, axs = plt.subplots(2, 2, figsize=(30, 30))

        # Plot time domain signal
        axs[0][0].plot(time, time_domain_pred)
        axs[1][0].plot(time, x[0][0])

        axs[0][0].set_title("Predicted denoised eq")
        axs[1][0].set_title("Noisy eq input")

        signal_file, noise_file, snr, event_shift = val_assoc[0]

        print(f"Event shift: {event_shift}")
        print(f"Signal to noise ratio snr={snr}")

        eq = np.load(signal_file, allow_pickle=True)
        noise = np.load(noise_file, allow_pickle=True)['noise_waveform_Z'][:6120]
        
        Z_eq = eq["earthquake_waveform_Z"][event_shift : event_shift + 6120]

        axs[0][1].plot(time, Z_eq)
        axs[1][1].plot(time, noise)

        axs[0][1].set_title("Clean earthquake signal")
        axs[1][1].set_title("Noise")

        """
        # Plot masks
        cax0 = axs[0][1].imshow(
            predictions[0, 0, :, :], cmap="plasma", interpolation="none"
        )  # 'viridis' is a colormap
        cax1 = axs[1][1].imshow(
            ground_truth[0, 0, :, :], cmap="plasma", interpolation="none"
        )

        axs[0][0].set_title("Prediction time domain")
        axs[1][0].set_title("Ground Truth time domain")

        axs[0][1].set_title("Prediction stft mask")
        axs[1][1].set_title("Ground Truth stft mask")

        fig.colorbar(cax0, ax=axs[0][1])
        fig.colorbar(cax1, ax=axs[1][1])
        axs[0][1].invert_yaxis()
        axs[1][1].invert_yaxis()
        """

        plt.show()

        return predictions

    return np.zeros(1)

def fit_butterworth(args: Namespace) -> dict:

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def vectorized_bandpass_filter(data, lowcut, highcut, fs, order=5):
        # Apply the bandpass filter along each row (axis=1) for 2D arrays or across the entire array if 1D
        return np.apply_along_axis(
            apply_bandpass_filter,
            axis=-1,
            arr=data,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            order=order,
        )

    signal_df = pd.read_pickle(args.signal_path)
    noise_df = pd.read_pickle(args.noise_path)

    signal = np.stack([np.array(x) for x in signal_df["Z"]])
    noise = np.stack([np.array(x) for x in noise_df["Z"]])

    n = len(signal)
    len_sample = 6000
    noise = noise[:n]
    signal_std = np.std(signal[:, 6000:6500], axis=1)
    noise_std = np.std(noise[:, 6000:6500], axis=1)
    snr_original = signal_std / noise_std
    noise_snr_mod = noise * snr_original[:, np.newaxis]  # rescale noise so that SNR=1
    snr_random = np.random.uniform(0.5, 2, n)  # random SNR
    event_snr_mod = signal * snr_random[:, np.newaxis]  # rescale event to desired SNR

    # event_shift = np.random.randint(1000,6000, n)
    start_pos = 3000
    ground_truth = event_snr_mod[:, start_pos : len_sample + start_pos]
    noisy_event = ground_truth + noise_snr_mod[:, :len_sample]

    def objective(params):
        print("filtering")
        lowcut, highcut, order = params
        order = int(round(order))
        filtered = vectorized_bandpass_filter(noisy_event, lowcut, highcut, 100, order)
        mean = np.mean(np.square(filtered - ground_truth))
        print(mean)
        return mean * 1000

    # Initial guess for the parameters
    initial_guess = [5.0, 30.0, 4]

    # Define bounds for the parameters: (lowcut, highcut, order)
    bounds = [
        (0.1, 30.0),  # Lowcut bounds
        (30.0, 50.0),  # Highcut bounds
        (1, 6),
    ]  # Order bounds
    print("Start minimizing")
    result = scipy.optimize.minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
        options={"disp": True, "maxiter": 50},
    )

    # Display the optimized result
    print(
        f"Optimized lowcut: {result.x[0]}, highcut: {result.x[1]}, order: {result.x[2]}"
    )

    return result

def fit_deep_denoiser(args: Namespace) -> keras.Model:

    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="earthquake denoising",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": "DeepDenoiser Unet2D",
                "dataset": "SED dataset",
                "epochs": args.epochs,
            },
        )

    model = Unet2D()

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=args.path_model + "/model_at_epoch_{epoch}.keras"
        ), 
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
    ]

    if args.wandb:
        wandb_callbacks = [WandbMetricsLogger()]
        callbacks = callbacks + wandb_callbacks

    train_dl, val_dl = get_dataloaders(args.signal_path, args.noise_path, args.batch_size)

    model.fit(
        train_dl, epochs=args.epochs, validation_data=val_dl, callbacks=callbacks
    )

    model.evaluate(val_dl, batch_size=32, verbose=2, steps=1)

    wandb.finish()

    return model



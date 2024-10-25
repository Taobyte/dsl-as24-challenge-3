import keras
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
import scipy.optimize
from tqdm import tqdm

from argparse import Namespace

import wandb
from wandb.integration.keras import WandbMetricsLogger

from src.models.DeepDenoiser.deep_denoiser_model import Unet2D
from src.data import get_dataloaders, get_signal_noise_assoc, InputSignals
from src.utils import Mode
from src.metrics import (
    cross_correlation,
    max_amplitude_difference,
    p_wave_onset_difference,
)

from obspy.signal.filter import bandpass


def train_model(args: Namespace) -> keras.Model:

    if args.butterworth:

        params = fit_butterworth(args)
        print(params)
        return params

    elif args.deepdenoiser:

        model = fit_deep_denoiser(args)
        return model

    return None


def compute_metrics(size_testset: int) -> pd.DataFrame:

    signal_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal"
    noise_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise"
    result_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/predictions/DeepDenoiser/"
    model_path = "C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/models/DeepDenoiser/model_at_epoch_3.keras"
    model_name = "deep_denoiser_metrics_test.csv"
    batch_size = 3

    snrs = [0.1 * i for i in range(1, 11)]
    signal_length = 6120

    assoc = get_signal_noise_assoc(signal_path, noise_path, Mode.TEST, size_testset)

    length_test_dataset = len(assoc)

    eq_traces = []
    noise_traces = []
    shifts = []

    for eq_path, noise_path, _, event_shift in assoc:

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
        shifts.append(event_shift)

    eq_traces = np.array(eq_traces)
    noise_traces = np.array(noise_traces)
    shifts = np.array(shifts)

    predictions = {
        "cc_mean": [],
        "cc_std": [],
        "max_amp_diff_mean": [],
        "max_amp_diff_std": [],
        "p_wave_mean": [],
        "p_wave_std": [],
    }

    model = keras.saving.load_model(model_path)
    for snr in tqdm(snrs, total=len(snrs)):
        test_dataset = InputSignals(assoc, Mode.TEST, snr)
        test_dl = DataLoader(test_dataset, batch_size, shuffle=False)
        masks = []
        for batch in test_dl:
            predicted_mask = model(batch)
            masks.append(predicted_mask)

        results = []
        for input_batch, mask_batch in zip(test_dl, masks):
            stft_real, stft_imag = keras.ops.stft(input_batch, 100, 24, 126)
            stft = np.concatenate([stft_real, stft_imag], axis=1)
            masked = stft * mask_batch
            time_domain_result = keras.ops.istft(
                (masked[:, :3, :, :], masked[:, 3:6, :, :]), 100, 24, 126
            )
            results.append(time_domain_result)

        denoised_eq = np.concatenate(results, axis=0)

        assert len(denoised_eq) == length_test_dataset

        # cross, SNR, max amplitude difference
        cross_correlations = np.array(
            [cross_correlation(a, b) for a, b in zip(eq_traces, denoised_eq)]
        )
        cross_correlation_mean = np.mean(cross_correlations)
        cross_correlation_std = np.std(cross_correlations)
        max_amplitude_differences = np.array(
            [max_amplitude_difference(a, b) for a, b in zip(eq_traces, denoised_eq)]
        )
        max_amplitude_difference_mean = np.mean(max_amplitude_differences)
        max_amplitude_difference_std = np.std(max_amplitude_differences)
        p_wave_onset_differences = np.array(
            [
                p_wave_onset_difference(a, b, shift)
                for a, b, shift in zip(eq_traces, denoised_eq, shifts)
            ]
        )
        p_wave_onset_difference_mean = np.mean(p_wave_onset_differences)
        p_wave_onset_difference_std = np.std(p_wave_onset_differences)

        predictions["cc_mean"].append(cross_correlation_mean)
        predictions["cc_std"].append(cross_correlation_std)
        predictions["max_amp_diff_mean"].append(max_amplitude_difference_mean)
        predictions["max_amp_diff_std"].append(max_amplitude_difference_std)
        predictions["p_wave_mean"].append(p_wave_onset_difference_mean)
        predictions["p_wave_std"].append(p_wave_onset_difference_std)

    df = pd.DataFrame(predictions)
    df["snr"] = snrs

    df.to_csv(result_path + model_name)

    return df


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
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    if args.wandb:
        wandb_callbacks = [WandbMetricsLogger()]
        callbacks = callbacks + wandb_callbacks

    train_dl, val_dl = get_dataloaders(
        args.signal_path, args.noise_path, args.batch_size
    )

    model.fit(train_dl, epochs=args.epochs, validation_data=val_dl, callbacks=callbacks)

    model.evaluate(val_dl, batch_size=32, verbose=2, steps=1)

    wandb.finish()

    return model

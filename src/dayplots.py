import torch
import obspy
import hydra
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from obspy.clients.fdsn import Client
from torch.utils.data import DataLoader

from models.ColdDiffusion.train_validate import load_model_and_weights
from models.ColdDiffusion.utils.testing import direct_denoising


def plot_day(cfg):
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    client = Client("ETH")
    utc = obspy.UTCDateTime(2019, 11, 6, 7, 0, 0, 0)
    print("Day of year (test data set ending on XX0): " + str(utc.julday))
    time_window = [14400, 14400]
    buffer = 100

    # get obspy stream = list of several obspy traces (1D arrays) with metainformation
    # # CH network DIX station channel/sensortype HH see https://www.fdsn.org/networks/detail/CH/
    print("Downloading data...")
    st_ch_dix = client.get_waveforms(
        network="CH",
        station="DIX",
        location="*",
        channel="HH*",
        starttime=utc - time_window[0] - buffer,
        endtime=utc + time_window[1] + buffer,
        attach_response=True,
    ).merge()
    # st_ch_dix = client.get_waveforms(network="CH",station="SIOO",location='*',channel='HG*',
    #                              starttime=utc -time_window[0]-buffer,endtime=utc +time_window[1] + buffer,attach_response=True).merge()

    cat = client.get_events(
        starttime=utc - time_window[0] - buffer, endtime=utc + time_window[1] + buffer
    )  # ETH earthquake catalogue

    st_ch_dix.remove_response(
        output="VEL", pre_filt=[1 / 1000, 1 / 200, 45, 50], water_level=None
    )  # deconvolve instrument response, go to physical units m/s
    if (
        st_ch_dix[0].stats.sampling_rate > 100
    ):  # downsample to 100sps, pre_filter removes >50Hz
        st_ch_dix.resample(100)  # pre filter

    st_ch_dix.trim(
        utc - time_window[0], endtime=utc + time_window[1]
    )  # remove buffer/edges

    y_limit = np.percentile(
        st_ch_dix.select(component="Z")[0].data, 99.95
    )  # use same y scale on raw and bandpass data
    fig = st_ch_dix.select(component="Z").plot(
        type="dayplot", vertical_scaling_range=y_limit, events=cat
    )
    fig.savefig(output_dir / "dayplot_raw.png")

    st_ch_dix_bandpass = st_ch_dix.copy()
    st_ch_dix_bandpass.filter("bandpass", freqmin=2, freqmax=30)
    fig = st_ch_dix_bandpass.select(component="Z").plot(
        type="dayplot", vertical_scaling_range=y_limit, events=cat
    )
    fig.savefig(output_dir / "dayplot_bandpass.png")

    # load model
    if cfg.model.model_name == "ColdDiffusion":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_and_weights(cfg.user.path_model, cfg)
        model = model.to(device)

    # create databatches
    st_ch_dix_denoised = st_ch_dix.copy()
    # optionally remove super low freq noise
    st_ch_dix_denoised.filter("bandpass", freqmin=0.01, freqmax=50)

    data_Z = st_ch_dix_denoised.select(component="Z")[0].data
    data_N = st_ch_dix_denoised.select(component="N")[0].data
    data_E = st_ch_dix_denoised.select(component="E")[0].data
    data = np.stack([data_Z, data_N, data_E], axis=1).T
    # optionally normalize data
    data = data / np.max(np.abs(data))

    # create batches
    signal_length = cfg.model.signal_length

    # Split the array into batches
    num_batches = int(np.floor(data.shape[1] / signal_length))
    chunks = np.array_split(data[:, : num_batches * signal_length], num_batches, axis=1)
    # ##### testing
    # filename = cfg.user.data.test_data_file
    # chunks = np.load(filename + "tst_noise_001.npy", allow_pickle=True)[:len(chunks)]

    dl = DataLoader(chunks, batch_size=32, shuffle=False)

    o = []
    for batch in tqdm(dl):
        if cfg.model.model_name == "ColdDiffusion":
            t = torch.Tensor([19]).to(device)
            pred = direct_denoising(model, batch.to(device).float(), t)

        o.extend(pred.cpu().numpy())

    o = np.concatenate(o, axis=1)
    st_ch_dix_denoised.select(component="Z")[0].data = o[0]
    st_ch_dix_denoised.select(component="N")[0].data = o[1]
    st_ch_dix_denoised.select(component="E")[0].data = o[2]
    y_limit = np.percentile(
        st_ch_dix_denoised.select(component="Z")[0].data, 99.95
    )  # use same y scale on raw and bandpass data
    fig = st_ch_dix_denoised.select(component="Z").plot(
        type="dayplot", vertical_scaling_range=y_limit, events=cat
    )
    fig.savefig(output_dir / "dayplot_denoised.png")

import torch
import torch.nn.functional as F
import einops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_istft(
    stft_eq: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    trace_length: int,
) -> torch.Tensor:
    window = torch.hann_window(win_length).to(device)
    stft_eq = einops.rearrange(stft_eq, "b (c f) w h -> (b c) w h f", c=3, f=2)
    stft_eq = stft_eq.contiguous()
    stft_eq = torch.view_as_complex(stft_eq)
    eq = torch.istft(
        stft_eq,
        n_fft,
        hop_length,
        win_length,
        window,
        length=trace_length,
        return_complex=False,
    )
    eq = einops.rearrange(eq, "(b c) t -> b c t", c=3)

    return eq


def get_stft(
    eq: torch.Tensor, n_fft: int, hop_length: int, win_length: int
) -> torch.Tensor:
    window = torch.hann_window(win_length).to(device)

    if len(eq.shape) == 2:
        eq_stft = torch.stft(
            eq,
            n_fft,
            hop_length,
            win_length,
            window,
            return_complex=True,
        )
        stft_eq = torch.view_as_real(eq_stft)
        stft_eq = einops.rearrange(stft_eq, "c w h f -> (c f) w h")

    elif len(eq.shape) == 3:
        B, C, T = eq.shape

        eq = einops.rearrange(eq, "b c t -> (b c) t", b=B, c=C)
        eq_stft = torch.stft(
            eq,
            n_fft,
            hop_length,
            win_length,
            window,
            return_complex=True,
        )
        stft_eq = torch.view_as_real(eq_stft)
        stft_eq = einops.rearrange(stft_eq, "(b c) w h f -> b (c f) w h", b=B, c=C, f=2)
    else:
        raise NotImplementedError

    return stft_eq


def get_mask(
    eq: torch.Tensor, noise: torch.Tensor, n_fft: int, hop_length: int, win_length: int
) -> torch.Tensor:
    stft_eq = get_stft(eq, n_fft, hop_length, win_length)
    stft_noise = get_stft(noise, n_fft, hop_length, win_length)

    mask = stft_eq.abs() / (stft_noise.abs() + stft_eq.abs() + 1e-12)

    return mask


# Adapted from https://github.com/kan-bayashi/ParallelWaveGAN

# Original Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    x_stft = torch.view_as_real(x_stft)

    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

        self.trace_length = 6120  # TODO: pass trace_length to LogSTFTMagnitudeLoss

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        # maybe add / self.trace_length
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        band="full",
        transform_stft=True,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band
        self.transform_stft = transform_stft

        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        if self.transform_stft:
            x_mag = stft(
                x, self.fft_size, self.shift_size, self.win_length, self.window
            )
            y_mag = stft(
                y, self.fft_size, self.shift_size, self.win_length, self.window
            )
        else:
            x_mag = x
            y_mag = y

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2  # only select high frequency bands
            sc_loss = self.spectral_convergence_loss(
                x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :]
            )
            mag_loss = self.log_stft_magnitude_loss(
                x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :]
            )
        elif self.band == "full":
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        else:
            raise NotImplementedError

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        sc_lambda=0.1,
        mag_lambda=0.1,
        band="full",
        transform_stft=True,
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            *_lambda (float): a balancing factor across different losses.
            band (str): high-band or full-band loss

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda
        self.transform_stft = transform_stft

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, band, transform_stft)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if self.transform_stft and len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)

        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss *= self.sc_lambda
        sc_loss /= len(self.stft_losses)
        mag_loss *= self.mag_lambda
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

import torch
import einops

from torch import nn, Tensor


class DownsamplingBlock(nn.Module):
    def __init__(self, channel_dim: int, dropout: float):
        super().__init__()

        self.conv2d_1 = nn.Conv2d(
            channel_dim, 2 * channel_dim, 5, padding="same", bias=False
        )
        self.conv2d_2 = nn.Conv2d(
            2 * channel_dim, 2 * channel_dim, 3, padding="same", bias=False
        )
        self.max_pooling = nn.MaxPool2d(2, stride=2)

        self.batch_norm1 = nn.BatchNorm2d(2 * channel_dim)
        self.batch_norm2 = nn.BatchNorm2d(2 * channel_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x) -> Tensor:
        x = self.conv2d_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout_1(x)

        x = self.conv2d_2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout_2(x)

        down = self.max_pooling(x)

        return down, x


class UpsamplingBlock(nn.Module):
    def __init__(self, channel_dim: int, dropout: float):
        super().__init__()

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_2d_1 = nn.ConvTranspose2d(
            4 * channel_dim, channel_dim, 5, padding=2, bias=False
        )
        self.conv_2d_2 = nn.ConvTranspose2d(
            channel_dim, channel_dim, 3, padding=1, bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(channel_dim)
        self.batch_norm2 = nn.BatchNorm2d(channel_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x, skip) -> Tensor:
        x = self.upsampling(x)
        x = torch.cat([x, skip], dim=1)

        x = self.conv_2d_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv_2d_2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x


class DeepDenoiser(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_layers: int,
        channel_base: int,
        dropout: float = 0.0,
        n_fft: int = 126,
        hop_length: int = 24,
        win_length: int = 100,
        window: str = "hann_window",
    ):
        super().__init__()

        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.inital_conv2d = nn.Conv2d(
            int(2 * n_channels), channel_base, kernel_size=3, stride=1, padding="same"
        )

        dims = [int(channel_base * 2**i) for i in range(n_layers)]

        self.encoder = nn.ModuleList([DownsamplingBlock(dim, dropout) for dim in dims])

        self.middle_conv = nn.Conv2d(
            channel_base * 2 ** (n_layers),
            channel_base * 2 ** (n_layers),
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.batch_norm = nn.BatchNorm2d(channel_base * 2 ** (n_layers))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        dims = list(reversed(dims))
        self.decoder = nn.ModuleList([UpsamplingBlock(dim, dropout) for dim in dims])

        self.final_conv2d = nn.Conv2d(
            channel_base, int(2 * n_channels), kernel_size=3, stride=1, padding="same"
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> Tensor:
        _, C, _ = x.shape
        x = einops.rearrange(x, "b c t -> (b c) t")
        x_stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            return_complex=False,
        )
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        x = torch.cat([real, imag], dim=0)
        x = einops.rearrange(x, "(b c) w h -> b c w h", c=int(2 * C))

        x = self.inital_conv2d(x)

        skip_connections = []
        for block in self.encoder:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.middle_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        skip_connections = list(reversed(skip_connections))
        for block, skip in zip(self.decoder, skip_connections):
            x = block(x, skip)

        x = self.final_conv2d(x)
        x = self.sigmoid(x)

        return x
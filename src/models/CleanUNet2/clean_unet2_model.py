import omegaconf
from torch import nn, Tensor
import einops

from src.utils import get_trained_model, Model
from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch


class CleanUNet2(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()

        self.trace_length = cfg.trace_length

        # Instantiate and load CleanSpecNet
        self.clean_specnet = get_trained_model(cfg, Model.DeepDenoiser)

        # Freeze CleanSpecNet's parameters
        for param in self.clean_specnet.parameters():
            param.requires_grad = False

        self.upsampling_block = nn.Sequential(
            nn.ConvTranspose2d(6, 6, kernel_size=(1, 4), stride=(1, 4)),
            nn.LeakyReLU(negative_slope=0.4),
            nn.ConvTranspose2d(6, 6, kernel_size=(1, 4), stride=(1, 4)),
            nn.LeakyReLU(negative_slope=0.4),
        )

        self.conv1x1 = nn.Conv1d(6 * 64, 3, 1, 1)

        self.clean_unet = CleanUNetPytorch(**cfg.model.clean_unet)

    def forward(self, noisy_eq) -> Tensor:
        clean_specnet_output = self.clean_specnet(noisy_eq)  # (B, 6, 64, 256)
        upsampled_spectogram = self.upsampling_block(
            clean_specnet_output
        )  # (B, 6, 64, 4096)

        # cropped = upsampled_spectogram[:, :, :, : self.trace_length]
        cropped_reshaped = einops.rearrange(
            upsampled_spectogram, "b c h w -> b (c h) w"
        )
        cropped_conved = self.conv1x1(cropped_reshaped)
        output = self.clean_unet(noisy_eq + cropped_conved)  # (B, 3, 4096)

        return output

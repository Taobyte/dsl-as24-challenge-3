import torch
from torch import nn, Tensor
import einops
import omegaconf

from src.models.CleanUNet.clean_unet_pytorch import CleanUNetPytorch
from src.models.DeepDenoiser.deep_denoiser_pytorch import DeepDenoiser


class CleanUnet2(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()

        # Instantiate and load CleanSpecNet
        self.clean_specnet = DeepDenoiser(**cfg.model.clean_specnet)
        checkpoint = torch.load(
            cfg.model.clean_specnet_path, map_location=torch.device("cpu")
        )
        self.clean_specnet.load_state_dict(checkpoint)

        # Freeze CleanSpecNet's parameters
        for param in self.clean_specnet.parameters():
            param.requires_grad = False

        self.upsampling_block = nn.Sequential(
            nn.ConvTranspose2d(6, 6, (3, 3), 5),
            nn.LeakyReLU(negative_slope=0.4),
            nn.ConvTranspose2d(6, 6, (3, 3), 5),
            nn.LeakyReLU(negative_slope=0.4),
        )

        self.conv1x1 = nn.Conv1d(6 * 1588, 3, 1, 1)

        self.clean_unet = CleanUNetPytorch(**cfg.model.clean_unet)

    def forward(self, noisy_eq) -> Tensor:
        clean_specnet_output = self.clean_specnet(noisy_eq)  # (B, 6, 64, 256)
        upsampled_spectogram = self.upsampling_block(
            clean_specnet_output
        )  # (B, 6, 1588, 6388)

        cropped = upsampled_spectogram[:, :, :, :6120]
        cropped_reshaped = einops.rearrange(cropped, "b c h w -> b (c h) w")
        cropped_conved = self.conv1x1(cropped_reshaped)
        output = self.clean_unet(noisy_eq + cropped_conved)  # (B, 3, 6120)

        return output

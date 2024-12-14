import torch
import torch.nn as nn
import einops

from src.models.CleanUNet.clean_unet_pytorch import TransformerEncoder, padding, weight_scaling_init
from src.models.CleanUNet.stft_loss import stft


class CleanSpecNet(nn.Module):
    """CleanSpecNet implementation"""

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4,
                 tsfm_n_layers=3, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048,
                 fft_size=126, 
                 hop_size=24, 
                 win_length=100, 
                 window="hann_window"):
        
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """

        super(CleanSpecNet, self).__init__()

        # Encoder parameters
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size

        # Transformer parameters
        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # STFT parameters
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        channels_input = int(channels_input * (fft_size // 2 + 1))

        self.initial_conv = nn.Conv1d(channels_input, channels_H, 1)

        # encoder and decoder
        self.encoder = nn.ModuleList()

        for i in range(encoder_n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(channels_H, channels_H, kernel_size, padding="same"),
                nn.ReLU(),
                nn.Conv1d(channels_H, channels_H * 2, kernel_size, padding="same"), 
                nn.GLU(dim=1)
            ))
        
        # self attention block
        self.tsfm_conv1 = nn.Conv1d(channels_H, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(d_word_vec=tsfm_d_model, 
                                               n_layers=tsfm_n_layers, 
                                               n_head=tsfm_n_head, 
                                               d_k=tsfm_d_model // tsfm_n_head, 
                                               d_v=tsfm_d_model // tsfm_n_head, 
                                               d_model=tsfm_d_model, 
                                               d_inner=tsfm_d_inner, 
                                               dropout=0.0, 
                                               n_position=0, 
                                               scale_emb=False)
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_H, kernel_size=1)
        channels_output = int(channels_output * (fft_size // 2 + 1))
        self.output_conv = nn.Conv1d(channels_H, channels_output, kernel_size=1)

        # weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        B, C, L = noisy_audio.shape # (B, 3, 6120)

        noisy_audio = einops.rearrange(noisy_audio, "b c t -> (b c) t")

        # compute STFT 
        x = stft(noisy_audio, self.fft_size, self.hop_size, self.win_length, self.window) # (3*B, #frames, fft_size // 2 + 1)
        x = einops.rearrange(x, "(repeat b) t c -> b (repeat c) t", repeat=self.channels_input)
    
        x = self.initial_conv(x)
        # encoder
        for encoder_block in self.encoder:
            x = encoder_block(x)

        # attention mask for causal inference; for non-causal, set attn_mask to None
        attn_mask = None

        x = self.tsfm_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)
        x = self.output_conv(x)

        x = x[:, :, :L]
        x = einops.rearrange(x, "b (repeat c) t -> b repeat c t", repeat=self.channels_output)
        x = torch.abs(x)
        
        return x


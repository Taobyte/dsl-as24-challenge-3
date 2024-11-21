import torch 

from src.models.CleanUNet.clean_unet_pytorch import CleanUNet
import einops


class CleanUnet2(torch.nn.Model):

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=3, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048):
        

    
    def forward(self, x):
        pass

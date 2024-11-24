import torch 
import einops

from src.models.CleanUNet.clean_unet_pytorch import CleanUNet
from src.models.CleanUNet2.clean_specnet import CleanSpecNet

class CleanUnet2(torch.nn.Model):

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=3, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048,
                 clean_specnet_path:str=None
                 ):
        
        super().__init__()
        
        # instantiate and load CleanSpecNet
        self.clean_specnet = CleanSpecNet()
        checkpoint = torch.load(clean_specnet_path, map_location=torch.device('gpu'))
        self.clean_specnet.load_state_dict(checkpoint['model_state_dict'])

        self.upsampling_block = torch.nn.Sequential([
            torch.nn.ConvTranspose2d(),
            torch.nn.ConvTranspose2d()
        ])

        self.clean_unet = CleanUNet()

    
    def forward(self, noisy_audio):
        
        clean_specnet_output = self.clean_specnet(noisy_audio) # (B, 3, 64, 256)
        upsampled_spectogram = self.upsampling_block(clean_specnet_output) # (B, 3, 6120)

        output = self.clean_unet(noisy_audio + upsampled_spectogram) # (B, 3, 6120)

        return output


        


from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class Model(Enum):
    Butterworth = "butter_worth"
    DeepDenoiser = "deep_denoiser"
    WaveDecompNet = "wave_decomp_net"
    ColdDiffusion = "cold_diffusion"
    CleanUNet = "clean_unet"
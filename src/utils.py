
from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class Model(Enum):
    Butterworth = "butter_worth"
    DeepDenoiser = "deep_denoiser"
    ColdDiffusion = "cold_diffusion"
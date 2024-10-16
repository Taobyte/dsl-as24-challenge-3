import numpy as np 
import os 
import keras 
from numpy import ndarray

class STFTLayer(keras.layers.Layer):
    
    def __init__(self, sequence_length: int, sequence_stride: int, fft_length: int, name=None):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length

    def call(self, x: ndarray) -> ndarray:

        # input shape: x.shape == (6000,3)
        # output shape: stft.shape == (256,256,6)

        stft_normalized = keras.ops.stft(
            x, self.sequence_length, self.sequence_stride, self.fft_length, window="hann", center=True
            ) 
        
        stft = None
        
        return stft_normalized, stft
    

class ISTFTLayer(keras.layers.Layer):
    def __init__(self, sequence_length: int, sequence_stride: int, fft_length: int, name=None):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length

    def call(self, x: ndarray) -> ndarray:

        # input shape: x.shape == (256,256,6)
        # output shape == (6000,3)
        
        return keras.ops.istft(
            x, self.sequence_length, self.sequence_stride, self.fft_length, window="hann", center=True
            )


class DownsamplingLayer(keras.layers.Layer):
    
    def __init__(self, channel_size: int, name=None):
        super().__init__(name=name)
        self.convoluation_layer = keras.Sequential(
            [
                keras.layers.Conv2D(channel_size, kernel_size=(3, 3), activation="relu", padding="same"),
                keras.layers.Conv2D(channel_size, kernel_size=(3, 3), activation="relu", padding="same"),
            ]
        )
        self.max_pooling_layer = keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, x):
        y = self.convoluation_layer(x)
        z = self.max_pooling_layer(y)
        return z, y

class UpsamplingLayer(keras.layers.Layer):
    
    def __init__(self, channel_size: int, name=None):
        super().__init__(name=name)
        self.transposed_conv = keras.layers.Conv2DTranspose(channel_size, kernel_size=3, strides=(2,2), activation='relu', padding="same")
        self.concatenate_layer = keras.layers.Concatenate(axis=-1)
        self.convolution_layer = keras.layers.Conv2D(channel_size, kernel_size=(3, 3), activation="relu", padding="same")

    def call(self, x: ndarray, skip_tensor: ndarray) -> ndarray:
        x = self.transposed_conv(x)
        x = self.concatenate_layer([x, skip_tensor])
        x = self.convolution_layer(x)
        return x
    

class UNET(keras.Model):
    def __init__(self):
        super().__init__()

        self.cl1 = DownsamplingLayer(16)
        self.cl2 = DownsamplingLayer(32)
        self.cl3 = DownsamplingLayer(64)
        self.cl4 = DownsamplingLayer(128)
        self.cl5 = DownsamplingLayer(256)

        self.inner_conv_block = keras.Sequential(
            [
                keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu"),
            ]
        )

        self.ul1 = UpsamplingLayer(256)
        self.ul2 = UpsamplingLayer(128)
        self.ul3 = UpsamplingLayer(64)
        self.ul4 = UpsamplingLayer(32)
        self.ul5 = UpsamplingLayer(16)

    def call(self, x: ndarray) -> ndarray:

        x, y1 = self.cl1(x)
        x, y2 = self.cl2(x)
        x, y3 = self.cl3(x)
        x, y4 = self.cl4(x)
        x, y5 = self.cl5(x)

        x = self.inner_conv_block(x)

        x = self.ul1(x,y1)
        x = self.ul2(x,y2)
        x = self.ul3(x,y3)
        x = self.ul4(x,y4)
        x = self.ul5(x,y5)

        return self.dense(x)
    

class MarsQNetwork(keras.Model):
   
   def __init__(self):
        super().__init__()

        self.stft_layer = STFTLayer(1,1,1)
        self.u_net = UNET()
        self.istft_layer = ISTFTLayer(1,1,1)

   
   def call(self, x: ndarray) -> tuple[ndarray, ndarray]:
       
       # input shape: x.shape == (6000,3)
       # output shape == (6000,3)
       
       stft_normalized, stft = self.stft_layer(x)
       mask_signal, mask_noise = self.u_net(stft_normalized)

       signal_stft = mask_signal * stft
       noise_stft = mask_noise * stft

       signal = self.istft_layer(signal_stft)
       noise = self.istft_layer(noise_stft)

       return signal, noise
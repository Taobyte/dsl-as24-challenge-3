import numpy as np 
import os 
import keras 
from numpy import ndarray


class DownsamplingLayer(keras.layers.Layer):
    
    def __init__(self, channel_size: int, conv_pool = True, name=None):
        super().__init__(name=name)

        self.rate = 0.0

        self.conv_pool = conv_pool

        self.conv = keras.layers.Conv2D(channel_size, kernel_size=(3, 3), activation="relu", padding="same", use_bias=False)
        self.batch_normalization_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()
        self.dropout_1 = keras.layers.Dropout(rate=self.rate)

        
        if self.conv_pool:
            self.pool = keras.layers.Conv2D(channel_size, kernel_size=(3, 3), strides=(2,2), activation="relu", padding="same", use_bias=False)
            self.batch_normalization_2 = keras.layers.BatchNormalization()
            self.relu_2 = keras.layers.ReLU()
            self.dropout_2 = keras.layers.Dropout(rate=self.rate)
        

    def call(self, x):

        x = self.conv(x)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)
        conv_output = self.dropout_1(x)

        if self.conv_pool:
            y = self.pool(conv_output)
            y = self.batch_normalization_2(y)
            y = self.relu_2(y)
            z = self.dropout_2(y)
            
            return z, conv_output
        
        return conv_output

class UpsamplingLayer(keras.layers.Layer):
    
    def __init__(self, channel_size: int, change_concat_dim = False, change_concat_dim_both = False, name=None):
        super().__init__(name=name)

        self.rate = 0.0
        self.change_concat_dim = change_concat_dim
        self.change_concat_dim_both = change_concat_dim_both

        self.transposed_conv = keras.layers.Conv2DTranspose(channel_size, kernel_size=3, strides=(2,2), activation=None, padding="same", use_bias= False)
        self.batch_normalization = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.drop_out = keras.layers.Dropout(rate=self.rate)

        self.concatenate_layer = keras.layers.Concatenate(axis=-1)

        self.convolution_layer = keras.layers.Conv2D(channel_size, kernel_size=(3, 3), activation=None, padding="same", use_bias=False)
        self.batch_normalization = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.drop_out = keras.layers.Dropout(rate=self.rate)

    def call(self, x: ndarray, skip_tensor: ndarray) -> ndarray:
        
        x = self.transposed_conv(x)

        if self.change_concat_dim:
            x = self.concatenate_layer([x[:,:, :-1,:], skip_tensor])
        elif self.change_concat_dim_both:
            x = self.concatenate_layer([x[:,:-1, :-1,:], skip_tensor])
        else:
            x = self.concatenate_layer([x, skip_tensor])
        
        x = self.convolution_layer(x)

        return x


class UNET(keras.Model):
    def __init__(self):
        super().__init__()

        self.cl0 = DownsamplingLayer(8)
        self.cl1 = DownsamplingLayer(16)
        self.cl2 = DownsamplingLayer(32)
        self.cl3 = DownsamplingLayer(64)
        self.cl4 = DownsamplingLayer(128)
        self.cl5 = DownsamplingLayer(256, conv_pool=False)

        self.ul0 = UpsamplingLayer(128, change_concat_dim=True)
        self.ul1 = UpsamplingLayer(64, change_concat_dim=False)
        self.ul2 = UpsamplingLayer(32, change_concat_dim=True)
        self.ul3 = UpsamplingLayer(16, change_concat_dim=True)
        self.ul4 = UpsamplingLayer(8, change_concat_dim_both=True)

        self.output_layer = keras.layers.Conv2D(2, kernel_size=1, activation=None, padding="same", use_bias=True)

    def call(self, x: ndarray) -> ndarray:
        
        x, y0 = self.cl0(x)
        x, y1 = self.cl1(x)
        x, y2 = self.cl2(x)
        x, y3 = self.cl3(x)
        x, y4 = self.cl4(x)
        x  = self.cl5(x)

        x = self.ul0(x,y4)
        x = self.ul1(x,y3)
        x = self.ul2(x,y2)
        x = self.ul3(x,y1)
        x = self.ul4(x,y0)

        output = self.output_layer(x)

        return output

   

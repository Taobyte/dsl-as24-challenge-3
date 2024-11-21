import keras
import math
# import keras_hub
import numpy as np
import einops

from src.models.CleanUNet.utils import CleanUNetInitializer, TransformerEncoder, GLUDown, GLUUp, RAGLUUp, RAGLUDown

class CleanUNet(keras.Model):
    """CleanUNet architecture."""

    def __init__(
        self,
        channels_input=3,
        channels_output=3,
        seq_length=6120,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=5,
        stride=2,
        tsfm_n_layers=3,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
        bottleneck="transformer",
        use_raglu=False,
        kernel_sizes=[9,7,5,3, 3, 3, 3, 3],
        name=None,
        **kwargs
    ):

        super(CleanUNet, self).__init__(name=name, **kwargs)

        assert len(kernel_sizes) == encoder_n_layers

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.seq_length = seq_length

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        self.bottleneck = bottleneck
        self.use_raglu = use_raglu

        self.encoder = []
        self.decoder = []

        initializer = CleanUNetInitializer(123)

        assert encoder_n_layers % 2 == 0
        mid = encoder_n_layers // 2

        for i in range(encoder_n_layers):
            
            if i < mid:
                self.encoder.append(GLUDown(channels_H, kernel_sizes[i], stride, initializer, name = f"GLUDown_{i}"))
            else:
                if use_raglu:
                    self.encoder.append(RAGLUDown(channels_H, kernel_sizes[i], stride, initializer, name = f"RAGLUDown_{i}"))
                else:
                    self.encoder.append(GLUDown(channels_H, kernel_sizes[i], stride, initializer, name = f"GLUDown_{i}"))
            

            channels_input = channels_H

            if i == 0:
                # no relu at end
                self.decoder.append(GLUUp(channels_H, channels_output, kernel_sizes[i], stride, initializer, False, name=f"GLUUp_{i}"))
            else:
                
                if i < mid:
                    self.decoder.insert(0, GLUUp(channels_H, channels_output, kernel_sizes[i], stride, initializer, True, f"GLUUp_{i}"))
                else:
                    if use_raglu:
                        self.decoder.insert(0, RAGLUUp(channels_H, channels_output, kernel_sizes[i], stride, initializer, True, f"RAGLUUp_{i}"))
                    else:
                        self.decoder.insert(0, GLUUp(channels_H, channels_output, kernel_sizes[i], stride, initializer, True, f"GLUUp_{i}"))
                
            channels_output = channels_H    
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # Transformer Bottleneck
        if self.bottleneck == "transformer":

            self.tsfm_conv1 = keras.layers.Conv1D(tsfm_d_model, kernel_size=1, kernel_initializer=initializer)
            self.tsfm_encoder = TransformerEncoder(
                d_word_vec=tsfm_d_model,
                n_layers=tsfm_n_layers,
                n_head=tsfm_n_head,
                d_k=tsfm_d_model // tsfm_n_head,
                d_v=tsfm_d_model // tsfm_n_head,
                d_model=tsfm_d_model,
                d_inner=tsfm_d_inner,
                dropout=0.0,
                n_position= int(np.ceil(seq_length / (2**encoder_n_layers))),
                scale_emb=False,
            )
            self.tsfm_conv2 = keras.layers.Conv1D(channels_output, kernel_size=1, kernel_initializer=initializer)
            print("Bottleneck Transformer")
        
        elif self.bottleneck == "lstm":

            self.lstm_layers = keras.Sequential([keras.layers.Bidirectional(keras.layers.LSTM(channels_output, return_sequences=True)) for _ in range(tsfm_n_layers)])
            print("Bottlneck LSTM")
        
        else:

            self.bottleneck_conv = keras.layers.Conv1D(channels_output, 1, activation="relu")
            print("Bottleneck Conv1D")


    def call(self, x):
        _, T, _ = x.shape
        
        # encoder
        skip_connections = []
        for downsampling_layer in self.encoder:
            x = downsampling_layer(x)
            skip_connections.append(x)
        
        skip_connections = skip_connections[::-1]

        if self.bottleneck == "transformer":
            x = self.tsfm_conv1(x)
            x = self.tsfm_encoder(x)
            x = self.tsfm_conv2(x)
        elif self.bottleneck == "lstm":
            x = self.lstm_layers(x)
        else:
            x = self.bottleneck_conv(x)

        # decoder
        for i, upsampling_layer in enumerate(self.decoder):
            skip = skip_connections[i]
            _, skip_T, _ = skip.shape
            x = keras.ops.concatenate([x[:, :skip_T, :],skip], axis=2)  # adding instead of concatenation
            x = upsampling_layer(x)

        x = x[:, :T, :]

        return x

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "channels_input": self.channels_input,
                "channels_output": self.channels_output,
                "channels_H": self.channels_H,
                "max_H": self.max_H,
                "encoder_n_layers": self.encoder_n_layers,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "seq_length": self.seq_length,
                "tsfm_n_layers": self.tsfm_n_layers,
                "tsfm_n_head": self.tsfm_n_head,
                "tsfm_d_model": self.tsfm_d_model,
                "tsfm_d_inner": self.tsfm_d_inner,
                "bottleneck": self.bottleneck,
                "use_raglu": self.use_raglu
            }
        )

        return config

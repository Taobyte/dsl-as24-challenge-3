import keras 

class Block(keras.layers.Layer):
    def __init__(self, units: int, dropout: float):
        super().__init__()
        self.dense = keras.layers.Dense(units, activation=None, use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization()
        self.act_fn = keras.layers.ReLU()
        self.dropout = keras.layers.Dropout(dropout)
    def call(self, x):
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MLP(keras.Model):

    def __init__(self, input_dim: int, hidden_units: list[int], dropout:float=0.0, name="mlp"):
        super().__init__()

        self.hidden_layers = keras.Sequential([Block(units, dropout) for units in hidden_units])
        self.output_layer = keras.layers.Dense(input_dim, activation=None)

    def call(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x




import numpy as np

from src.models.DummyModel.mlp import MLP

def test_mlp():

    mlp = MLP(10, [5], 0.1)
    input = np.zeros((32,1,10))
    output = mlp(input)

    assert output.shape == input.shape
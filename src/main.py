import config_parser as cp
import random
import numpy as np
import tensorflow as tf
import torch as th
import jax
import os
from train_validate import train_model, test_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "jax"

args = cp.configure_args()

print(f'Signal folder path: {args.signal_path}')
print(f'Noise folder path{args.noise_path}')
print(f'Seed: {args.seed}')

# Set see (default: 123)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
th.manual_seed(args.seed)
jax_key = jax.random.PRNGKey(args.seed)


if args.training:
    model = train_model(args)
else:
    loss = test_model(args)

import config_parser as cp
import os
from train_validate import train_model, test_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "jax"

args = cp.configure_args()

print(args.signal_path)
print(args.noise_path)

if args.training:
    model = train_model(args)
else:
    loss = test_model(args)

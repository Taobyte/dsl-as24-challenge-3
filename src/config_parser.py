import argparse
import os

def configure_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # signal path
    parser.add_argument("--signal_path",
                    default="C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/signal",
                    type=str,
                    help="Path to earthquake signal folder")
    # signal path
    parser.add_argument("--noise_path",
                    default="C:/Users/cleme/ETH/Master/DataLab/dsl-as24-challenge-3/data/noise",
                    type=str,
                    help="Path to noise folder")

    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="How many GPUs used for training (default: 0)")
    
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="Learning rate for the optimizer (default: 0.001)")
    
    parser.add_argument("--dropout_rate",
                        default=0.0,
                        type=float,
                        help="Learning rate for the optimizer (default: 0.0)")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size (default: 32)")
    
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Number of training epochs (default: 10)")
    
    parser.add_argument("--seq_length",
                        default=3000,
                        type=int,
                        help="Length of the earthquake signal (default: 3000)")
    
    parser.add_argument("--training",
                    default=False,
                    type=bool,
                    help="Whether to train or test the model (default: False)")
    
    parser.add_argument("--seed",
                        default=123,
                        type=int,
                        help="Length of the earthquake signal (default: 123)")
    
    parser.add_argument("--length_dataset",
                        default=None,
                        type=int,
                        help="Specifies how many training samples to use")

    parser.add_argument("--checkpoint_path",
                        default="./models",
                        type=str,
                        help="Path to save checkpoints (default: ./models)")
    
    parser.add_argument("--file_name",
                        default="DeepDenoiser",
                        type=str,
                        help="File name for saved models (default: DeepDenoiser)")
    
    parser.add_argument("--path_model",
                    default="./models",
                    type=str,
                    help="Path to the model weights (default: full path)")
    
    # Parse arguments
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    return args
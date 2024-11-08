import numpy as np
import keras
from torch.utils.data import DataLoader, Subset

from src.models.CleanUNet.dataset import CleanUNetDataset
from src.models.CleanUNet.clean_unet_model import CleanUNet, CleanUNetLoss

def test_clean_unet_serializability():

    signal_path = '/cluster/scratch/ckeusch/data/signal/train/'
    noise_path = '/cluster/scratch/ckeusch/data/noise/train/'
    indices = np.arange(32)
    train_dl = DataLoader(Subset(CleanUNetDataset(signal_path, noise_path, 6120), indices), batch_size=32)
    print(f'lenght train_dl = {len(train_dl)}')
    model = CleanUNet()

    sample_shape = np.zeros(
        (32, 6120, 3)
    )
    model(sample_shape)

    model.compile(
        loss=CleanUNetLoss(),
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
    )

    callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="./model_at_epoch_{epoch}.keras"
            ),
        ]
    epochs = 1
    model.fit(train_dl, callbacks=callbacks, epochs=epochs)

    new_model = keras.saving.load_model("./model_at_epoch_1.keras")

    assert True
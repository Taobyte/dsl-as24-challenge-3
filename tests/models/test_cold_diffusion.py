import numpy as np
import keras
from torch.utils.data import DataLoader, Subset

from src.data import TimeDomainDataset
from src.models.ColdDiffusion.cold_diffusion_model_clemens import Unet1D
from src.utils import Model



def test_cold_diffusion_config_serializability():

    signal_path = '/cluster/scratch/ckeusch/data/signal/train/'
    noise_path = '/cluster/scratch/ckeusch/data/noise/train/'
    indices = np.arange(32)
    train_dl = DataLoader(Subset(TimeDomainDataset(signal_path, noise_path, 2048, model = Model.ColdDiffusion), indices), batch_size=32)
    print(f'lenght train_dl = {len(train_dl)}')
    model = Unet1D(
        8,
        8,
        dim_mults=(1,2,4,8),
        channels=3,
    )

    sample_shape = np.zeros(
        (32, 2048, 3)
    )
    model(sample_shape)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
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



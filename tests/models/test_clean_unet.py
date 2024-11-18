import numpy as np
import matplotlib.pyplot as plt
import keras
import math
from torch.utils.data import DataLoader, Subset


from src.models.CleanUNet.dataset import CleanUNetDataset
from src.models.CleanUNet.utils import CleanUNetLoss
from src.models.CleanUNet.clean_unet_model import CleanUNet

def test_clean_unet_serializability():

    keras.utils.set_random_seed(123)

    signal_path = '/cluster/scratch/ckeusch/data/signal/train/'
    noise_path = '/cluster/scratch/ckeusch/data/noise/train/'
    indices = np.arange(32)
    dataset = CleanUNetDataset(signal_path, noise_path, 6120, random=False)
    train_dl = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False)
    print(f'lenght train_dl = {len(train_dl)}')
    model = CleanUNet(bottleneck=None, encoder_n_layers=8, use_raglu=False, channels_H=8, tsfm_n_layers=3)

    sample_shape = np.zeros(
        (32, 6120, 3)
    )
    model(sample_shape)

    def scheduler(epoch, lr):
        if epoch < 14:
            return lr
        else:
            return lr * math.exp(-0.005)
    
    callback = keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )

    # callbacks = [callback]
    callbacks = []
    epochs = 1000
    history = model.fit(train_dl, callbacks=callbacks, epochs=epochs)

    plt.plot(history.history['loss'])
    plt.title('Model Loss During Overfitting')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig('overfit_training_loss.png')

    loss = model.evaluate(train_dl)
    assert loss <= 0.05


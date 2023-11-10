import pytorch_lightning as L

from src.cifar10 import CIFAR10DataModule
from src.net import LitModel


if __name__ == '__main__':
    dm = CIFAR10DataModule()
    model = LitModel()
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model, dm)

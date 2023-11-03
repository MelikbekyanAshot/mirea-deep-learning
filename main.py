from config.constants import EPOCH_NUMBER
from model.net import Net
from training.trainer import Trainer
from data_import.cifar10 import train_loader, test_loader


if __name__ == '__main__':
    net = Net()
    trainer = Trainer(model=net)
    trainer.train(train_loader=train_loader, num_epochs=EPOCH_NUMBER)
    trainer.test(test_loader=test_loader)

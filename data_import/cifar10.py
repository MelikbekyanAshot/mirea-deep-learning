import torch
import torchvision
import torchvision.transforms as transforms

from config.constants import MEAN_NORM, STD_NORM, BATCH_SIZE, NUM_WORKERS


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(MEAN_NORM, STD_NORM)])


train_set = torchvision.datasets.CIFAR10(root='./data_import', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)

test_set = torchvision.datasets.CIFAR10(root='./data_import', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=NUM_WORKERS)

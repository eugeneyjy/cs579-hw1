import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REPO_DIR / 'datasets'

def data_loader(dataset, batch_size, transform, train=True, valid_size=0.2):
    datasets_dict = {
        'mnist': datasets.MNIST,
        'cifar10': datasets.CIFAR10
    }

    data = datasets_dict[dataset](
        root=DATASETS_DIR,
        train=train,
        transform=transform,
        download=True
    )

    if train:
        train_size = (int)((1 - valid_size) * len(data))
        valid_size = len(data) - train_size
        train_data, valid_data = random_split(data, [train_size, valid_size])

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True
        )

        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=batch_size,
            shuffle=True
        )

        return train_loader, valid_loader
    
    else:
        test_loader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True
        )

        return test_loader
    
def batch_mean_and_std(loader):
    mean = 0
    std = 0

    for images, _ in loader:
        mean += images.mean()
        std += images.std()
    
    return mean/len(loader), std/len(loader)


def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        # transforms.Normalize(mean=(0.1309,), std=(0.2893,))
    ])

    train_loader, valid_loader = data_loader('cifar10', 64, transform, train=True)
    # first_image = train_loader[0][0].squeeze()
    # plt.imshow(first_image, cmap='gray')
    # plt.show()

main()
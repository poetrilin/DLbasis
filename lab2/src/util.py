import os
import torch
from torchvision import datasets, transforms
from cutout import Cutout


def load_data(data_dir, batch_size):

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          Cutout(n_holes=1, length=16)])

    train_data = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform)

    train_data_size = int(len(train_data)*0.8)
    val_data_size = len(train_data)-train_data_size
    train_data, val_data = torch.utils.data.random_split(
        train_data, [train_data_size, val_data_size])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

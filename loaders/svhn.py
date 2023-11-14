import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

def get_svhn_loaders(root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    # train_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='train', download=True, transform=transform_train)
    # extra_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='extra', download=True, transform=transform_train)
    dev_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='train', download=True, transform=transform_train)
    test_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='test', download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [72257,1000])
    torch.set_rng_state(seed)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader
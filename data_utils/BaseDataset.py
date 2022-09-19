import os
import copy
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import Dataset, DataLoader, sampler, Subset
from utils import *

__all__ = ['BaseDataset']

DATASET = {'dirichlet-mnist': MNIST, 'dirichlet-cifar10': CIFAR10,
           'dirichlet-fmnist': FashionMNIST, 'dirichlet-cifar100': CIFAR100}

def mean_list(data):
    if data == 'dirichlet-cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        img_size = 32
    elif data == 'dirichlet-cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        img_size = 32
    elif data == 'dirichlet-mnist':
        mean = [0.5]
        std = [0.5]
        img_size = 28
    elif data == 'dirichlet-fmnist':
        mean = [0.5]
        std = [0.5]
        img_size = 28
    else:
        raise Exception('No option for this dataset.')

    return mean, std, img_size

def transform_setter(dataset, train=True):
    mean, std, img_size = mean_list(dataset)
    
    if dataset == 'dirichlet-mnist' or dataset == 'dirichlet-fmnist':
        train_transforms = [transforms.RandomResizedCrop((img_size, img_size))]
        test_transforms = [transforms.Resize((img_size, img_size))]
    else:
        train_transforms = [transforms.RandomCrop(img_size, padding=4), transforms.RandomHorizontalFlip()]
        test_transforms = []
        
    train_transforms = transforms.Compose(train_transforms + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transforms = transforms.Compose(test_transforms + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    
    if train:
        return train_transforms
    else:
        return test_transforms

class BaseDataset(Dataset):
    def __init__(self, args, train=True, noisy=True):
        self.agrs = args
        self.data = args.dataset
        self.train=train
        self.datadir = os.path.join(args.data_dir, self.data)
        self.transforms = transform_setter(self.data, train)
        self.noisy = noisy
        self.noise = {'ratio': args.noise, 'type': args.noise_type}
        self.device = args.device

        os.makedirs(self.datadir, exist_ok=True)
        self.dataset = DATASET[self.data](self.datadir, train=self.train, transform = self.transforms, download = train)
        self.len = len(self.dataset)
        self.classes = len(self.dataset.classes)
        self.gt_labels = copy.deepcopy(self.dataset.targets)
        if self.noisy and self.noise['ratio'] > 0.0:
            self.make_noise()
    
    def make_noise(self):
        if self.noise['type'] == 'sym':
            indices = np.random.permutation(self.len)
            for i in range(int(self.noise['ratio'] * self.len)):
                self.dataset.targets[indices[i]] = np.random.randint(self.classes)
        else:
            raise NotImplementedError("developing...")
        return 

    def __getitem__(self, index):
        img, label = self.dataset[index]
        gt_label = self.gt_labels[index]
        return index, img, label, gt_label
    
    def __len__(self):
        return self.len
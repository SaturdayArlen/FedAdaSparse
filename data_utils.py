import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar_loaders(dataset_name='cifar10', batch_size=64, num_workers=2,
                      data_path=None, normalize='cifar'):
    if dataset_name == 'cifar10':
        num_classes = 10
        if normalize == 'imagenet':
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                                transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                               transform=transform_val)
    elif dataset_name == 'cifar100':
        num_classes = 100
        if normalize == 'imagenet':
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True,
                                                  transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True,
                                               transform=transform_val)
    elif dataset_name == 'fashionmnist':
        num_classes = 10
        if normalize == 'imagenet':
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = (0.2860,), (0.3530,)
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True,
                                                     transform=transform_train)
        valset = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True,
                                                    transform=transform_val)
    elif dataset_name == 'svhn':
        num_classes = 10
        if normalize == 'imagenet':
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = torchvision.datasets.SVHN(root='./dataset', split='train', download=True,
                                              transform=transform_train)
        valset = torchvision.datasets.SVHN(root='./dataset', split='test', download=True,
                                            transform=transform_val)
    elif dataset_name == 'tiny_imagenet':
        num_classes = 200
        if data_path is None:
            data_path = './dataset/tiny-imagenet-200'
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
        valset = torchvision.datasets.ImageFolder(root=valdir, transform=transform_val)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, trainset, valset, num_classes


def get_dataset_loaders_224(dataset_name, batch_size_test, normalize='imagenet'):
    if normalize == 'imagenet':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    if dataset_name == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                                transform=transform_test)
    elif dataset_name == 'cifar100':
        num_classes = 100
        if normalize != 'imagenet':
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True,
                                                  transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True,
                                                 transform=transform_test)
    elif dataset_name == 'fashionmnist':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True,
                                                     transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True,
                                                    transform=transform_test)
    elif dataset_name == 'svhn':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = torchvision.datasets.SVHN(root='./dataset', split='train', download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.SVHN(root='./dataset', split='test', download=True,
                                            transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainset, testset, num_classes

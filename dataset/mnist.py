import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms


def get_mnist_dataloader(target, bs=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    trainset = NumInMNIST(root=data_path, train=True, download=True, transform=transform, target_number=target)
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = NumInMNIST(root=data_path, train=False, download=True, transform=transform, target_number=target)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=True)

    return train_loader, test_loader


def get_mnist_fs_dataloader(target, bs=4, data_path="./data", transform=None, num_sample=10):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    trainset = FewShotInMNIST(root=data_path, train=True, download=True, transform=transform,
                              target_number=target, num_sample=num_sample)
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = FewShotInMNIST(root=data_path, train=False, download=True, transform=transform,
                             target_number=target, num_sample=num_sample)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=True)

    return train_loader, test_loader


class NumInMNIST(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 target_number=None,
                 ):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        super().__init__(root, train, transform, target_transform, download)
        if target_number is not None:
            class_data = []
            class_target = []
            for i in range(len(self.targets)):
                if self.targets[i] == target_number:
                    class_data.append(self.data[i])
                    class_target.append(self.targets[i])
            self.data = class_data
            self.targets = class_target


class FewShotInMNIST(NumInMNIST):
    def __init__(self,
                 root: str,
                 target_number: int, num_sample=10, sampled_idx=None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, train, transform, target_transform, download, target_number)

        total_len = len(self.data)
        if sampled_idx is not None:
            assert num_sample == len(sampled_idx)
        else:
            sampled_idx = np.random.choice(total_len, min(total_len, num_sample), replace=False)
        new_data, new_targets = [], []
        for idx in sampled_idx:
            new_data.append(self.data[idx])
            new_targets.append(self.targets[idx])
        self.data = new_data
        self.targets = new_targets


if __name__ == "__main__":

    ccf = FewShotInMNIST("../data/",  train=False, download=True, target_number=9)
    print(ccf.targets)
    print(len(ccf.targets))

import os
import numpy as np
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms

from dataset.cifar10 import FewShotInCifar10
from dataset.mnist import FewShotInMNIST


class Tasks4MNIST:
    def __init__(self,
                 #
                 root: str, batch_size=16,
                 num_samples=10, num_tasks=30,
                 # cifar10 config:
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 target_number=0,
                 ):
        self.root = root
        self.num_samples = num_samples
        self.num_tasks = num_tasks
        self.transform = transform
        self.download = download
        self.target_number = target_number
        self.batch_size = batch_size
        self.CIFAR10_CLASS_DATA_NUM = 1000

    def __len__(self):
        return self.num_tasks

    def __iter__(self):
        while True:
            sampled_idx = np.random.choice(self.CIFAR10_CLASS_DATA_NUM, (2, self.num_samples), replace=False)
            train_dataset = FewShotInMNIST(root=self.root, transform=self.transform, download=self.download,
                                           sampled_idx=sampled_idx[0], target_number=self.target_number)

            test_dataset = FewShotInMNIST(root=self.root, transform=self.transform, download=self.download,
                                          sampled_idx=sampled_idx[1], target_number=self.target_number)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            yield train_dataloader, test_dataloader


class Tasks4Cifar10:
    def __init__(self,
                 #
                 root: str, batch_size=16,
                 num_samples=10, num_tasks=30,
                 # cifar10 config:
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 class_name="",
                 ):
        self.root = root
        self.num_samples = num_samples
        self.num_tasks = num_tasks
        self.transform = transform
        self.download = download
        self.class_name = class_name
        self.batch_size = batch_size
        self.CIFAR10_CLASS_DATA_NUM = 1000

    def __len__(self):
        return self.num_tasks

    def __iter__(self):
        while True:
            sampled_idx = np.random.choice(self.CIFAR10_CLASS_DATA_NUM, (2, self.num_samples), replace=False)
            train_dataset = FewShotInCifar10(root=self.root, transform=self.transform, download=self.download,
                                             sampled_idx=sampled_idx[0], class_name=self.class_name)

            test_dataset = FewShotInCifar10(root=self.root, transform=self.transform, download=self.download,
                                            sampled_idx=sampled_idx[1], class_name=self.class_name)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            yield train_dataloader, test_dataloader


if __name__ == "__main__":
    taskloader = Tasks4Cifar10(root="../data/", download=True, class_name="cat", )
    it = iter(taskloader)
    train, test = next(it)
    print(len(taskloader))

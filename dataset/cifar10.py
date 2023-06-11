from torchvision.datasets import CIFAR10
import os
import numpy as np
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms


def get_cifar10_dataloader(class_name="", bs=4, img_size=32, data_path="./data", transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1)),
        ])

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    trainset = ClassInCifar10(root=data_path, train=True, download=True, transform=transform, class_name=class_name)
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = ClassInCifar10(root=data_path, train=False, download=True, transform=transform, class_name=class_name)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=True)

    return train_loader, test_loader


def get_cifar10_fs_dataloader(class_name="", bs=4, img_size=32, data_path="./data", transform=None,
                              num_sample=10):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0,), (1, 1, 1,)),
        ])

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    trainset = FewShotInCifar10(root=data_path, train=True, download=True, transform=transform,
                                num_sample=num_sample, class_name=class_name)
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = FewShotInCifar10(root=data_path, train=False, download=True, transform=transform,
                               num_sample=num_sample, class_name=class_name)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=True)

    return train_loader, test_loader


class ClassInCifar10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 class_name="",
                 ):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        super().__init__(root, train, transform, target_transform, download)

        if class_name != "":
            class_idx = self.class_to_idx[class_name]
            class_data = []
            class_target = []
            for i in range(len(self.targets)):
                if self.targets[i] == class_idx:
                    class_data.append(self.data[i])
                    class_target.append(self.targets[i])
            self.data = class_data
            self.targets = class_target


class FewShotInCifar10(ClassInCifar10):
    def __init__(self,
                 root: str,
                 class_name: str, num_sample=10, sampled_idx=None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, train, transform, target_transform, download, class_name)

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
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for cls in classes:

        ccf = ClassInCifar10("../data/",  train=False, download=True, class_name=cls)
        print(len(ccf.targets))

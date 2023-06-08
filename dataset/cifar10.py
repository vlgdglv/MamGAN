from torchvision.datasets import CIFAR10
import os
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms


def get_cifar10_dataloader(class_name="", bs=4, img_size=32, data_path="./data",transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    trainset = ClassInCifar10(root=data_path, train=True, download=True, transform=transform, class_name=class_name)
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = ClassInCifar10(root=data_path, train=False, download=True, transform=transform, class_name=class_name)
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


if __name__ == "__main__":

    ccf = ClassInCifar10("../data/",  train=False, download=True, class_name="automobile")
    print(ccf.classes)
    print(len(ccf.targets))

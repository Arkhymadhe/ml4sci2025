# -*- coding: utf-8 -*-

import os
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T

from random import shuffle

__all__ = [
    "AlzheimerData",
    "AlzheimerDataset",
    "MNISTDataset"
]

class AlzheimerData(Dataset):
    def __init__(self, path, train=True):
        super().__init__()

        self.path = path
        split = "train" if train else "test"

        path = path + "/" + split

        file_paths = [os.path.join(path, f) for f in os.listdir(path)]

        files = []

        for class_path in file_paths:
            fnames = os.listdir(class_path)
            fnames = [os.path.join(class_path, f) for f in fnames]
            files += fnames

        shuffle(files)

        self.files = files

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(128),
                T.Normalize(mean=.5, std=.5)
            ]
        )

        self.label_map = {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3
        }

    def __getitem__(self, index):
        fpath = self.files[index]
        class_label = fpath.split(os.sep)[-2]
        return self.transforms(Image.open(fpath)), self.label_map[class_label]

    def __len__(self):
        return len(self.files)

    @classmethod
    def from_folder(cls, path):
        return ConcatDataset([cls(path=path, train=True), cls(path=path, train=False)])

class AlzheimerDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = ConcatDataset(
            [
                AlzheimerData(path=path, train=True),
                AlzheimerData(path=path, train=False)
            ]
        )
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class MNISTDataset(Dataset):
    def __init__(self, input_size=64):
        super().__init__()

        from torchvision.datasets import MNIST

        self.input_size = input_size

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((input_size, input_size)),
                T.Normalize(mean=.5, std=.5)
            ]
        )

        A = MNIST(root='./data', train=True, download=True, transform=self.transforms)
        B = MNIST(root='./data', train=False, download=True, transform=self.transforms)

        self.dataset = ConcatDataset([A, B])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indx):
        return self.dataset[indx]
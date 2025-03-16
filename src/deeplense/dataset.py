# -*- coding: utf-8 -*-

import os
import numpy as np

from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.deeplense.models import DiffusionHelper

__all__ = [
    "PatchTransform",
    "MaskTransform",
    "DeepLenseSRDataset",
    "DeepLenseDiffusionDataset",
    "DeepLenseMaskedDataset"
]


class PatchTransform(nn.Module):
    def __init__(self, num_patches, image_size):
        super(PatchTransform, self).__init__()

        if isinstance(image_size, int):
            H, W = image_size, image_size
        else:
            H, W = image_size

        self.patch_height = int(H / (num_patches / 2))
        self.patch_width = int(W / (num_patches / 2))

        if self.patch_height == self.patch_width:
            self.patch_size = (self.patch_height, self.patch_width)
        else:
            self.patch_size = self.patch_width

    def forward(self, x):
        x = x.unfold(1, self.patch_height, self.patch_height)  # 3 x 8 x 32 x 4
        x = x.unfold(2, self.patch_width, self.patch_width)  # 3 x 8 x 8 x 4 x 4

        return x.reshape(-1, self.patch_width * self.patch_height * 3)  # -1 x 48 == 64 x 48

class MaskTransform(nn.Module):
    def __init__(
        self,
        num_patches: int,
        use_mask: bool = False,
        mask_token: int = -100,
        mask_ratio: float =.75
    ):
        super(MaskTransform, self).__init__()

        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.use_mask = use_mask
        self.mask_token = mask_token

    def mask_patches(self, patched_image: torch.Tensor) -> Sequence[torch.Tensor]:
        num_patches_to_mask = int(self.num_patches * self.mask_ratio)

        patch_indices = torch.distributions.Uniform(
            low=0, high=self.num_patches
        ).sample((num_patches_to_mask,))

        patch_mask = torch.ones(self.num_patches).bool()
        patch_mask[patch_indices] = False

        if self.use_mask:
            patched_image[~patch_mask, :] = self.mask_token
        else:
            patched_image = patched_image[patch_mask]

        return patched_image, patch_mask

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.mask_patches(x)

class DeepLenseSRDataset(Dataset):
    def __init__(
        self,
        path: Optional[str] = None,
        finetune: bool = False
    ):
        super().__init__()

        if path is None:
            if finetune:
                path = "../../data/deeplense_sr_extra_dataset"
            else:
                path = "../../data/deeplense_sr_dataset"

        self.path = path
        self.finetune = finetune

        self.high_resolution_files = [
            os.path.join(path, "Dataset", "HR", f)
            for f in os.listdir(os.path.join(path, "Dataset", "HR"))
        ]

        self.low_resolution_files = [
            os.path.join(path, "Dataset", "LR", f)
            for f in os.listdir(os.path.join(path, "Dataset", "LR"))
        ]

        self.tensor_transform = T.PILToTensor()

    def __len__(self) -> int:
        return len(self.low_resolution_files)

    @staticmethod
    def preprocess_image(path: str) -> torch.Tensor:
        return torch.tensor(np.load(path))

    def __getitem__(self, idx: int) -> [torch.Tensor]:
        x_hr = self.high_resolution_files[idx]
        x_lr = self.low_resolution_files[idx]

        return self.preprocess_image(x_lr), self.preprocess_image(x_hr)

    def to_dataloader(
        self,
        num_workers: int = 0,
        batch_size: int = 1,
        pin_memory: bool = True
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        return dataloader

    def __str__(self):
        return f"{self.__class__.__name__}(path = {self.path}, finetune = {self.finetune})"

class DeepLenseDiffusionDataset(Dataset):
    def __init__(
        self,
        diffusion_helper: DiffusionHelper = DiffusionHelper(),
        input_size: Optional[int, Sequence[int, int]] = 160,
        path: Optional[str] = None,
    ):
        super().__init__()

        if path is None:
            path = "../../data/deeplense_diffusion_dataset/Samples"

        self.path = path
        self.input_size = input_size
        self.diffusion_helper = diffusion_helper

        self.files = [
            os.path.join(path, f)
            for f in os.listdir(path)
        ]

        if self.input_size:
            t = T.Resize((self.input_size, self.input_size))
            t1 = T.RandomResizedCrop((self.input_size, self.input_size))
            t = T.Compose([t1, t])
        else:
            t = T.Lambda(lambda x: x)

        self.tensor_transform = T.Compose(
            [
                t,
                T.Normalize(mean = (.5,), std = (.5,))
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def preprocess_image(path: str) -> torch.Tensor:
        return torch.tensor(np.load(path))

    def __getitem__(self, idx: int) -> [torch.Tensor]:
        img = self.files[idx]

        return self.tensor_transform(self.preprocess_image(img))

    def to_dataloader(
        self,
        num_workers: int = 0,
        batch_size: int = 1,
        pin_memory: bool = True
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        return dataloader

    def __str__(self):
        return f"{self.__class__.__name__}(path = {self.path})"

class DeepLenseMaskedDataset(Dataset):
    def __init__(
        self,
        use_mask=False,
        num_patches=64,
        image_size = 224,
        mask_ratio=.75,
        path=None
    ):
        super().__init__()

        self.num_patches = num_patches
        self.use_mask = use_mask
        self.image_size = image_size
        self.mask_ratio = mask_ratio

        if path is None:
            path = "../../data/deeplense_sr_dataset"

        self.path = path

        self.high_resolution_files = [
            os.path.join(path, "Dataset", "HR", f)
            for f in os.listdir(os.path.join(path, "Dataset", "HR"))
        ]

        self.low_resolution_files = [
            os.path.join(path, "Dataset", "LR", f)
            for f in os.listdir(os.path.join(path, "Dataset", "LR"))
        ]

        self.tensor_transform = T.PILToTensor()

        self.patch_transform = T.Compose(
            [
                self.tensor_transform,
                PatchTransform(num_patches, image_size=image_size)
            ]
        )
        self.mask_transform = MaskTransform(
            num_patches=num_patches,
            mask_ratio=mask_ratio,
            use_mask=use_mask
        )

    def __len__(self):
        return len(self.low_resolution_files)

    @staticmethod
    def preprocess_image(path):
        return torch.tensor(np.load(path)).unsqueeze(-1)

    def __getitem__(self, idx):
        hr_fname = self.high_resolution_files[idx]
        lr_fname = self.low_resolution_files[idx]

        return self.preprocess_image(lr_fname), self.preprocess_image(hr_fname)

    def __str__(self):
        params = (f"use_mask = {self.use_mask}, num_patches = {self.num_patches},"
                  f" image_size = {self.image_size}, mask_ratio = {self.mask_ratio}")
        return f"{self.__class__.__name__}({params})"


if __name__ == "__main__":
    # patch_mask = torch.ones(10).bool()
    # print(patch_mask)
    # print(~patch_mask)

    finetune = True
    dataset = DeepLenseSRDataset(finetune=finetune)

    X, y = dataset[0]

    print(X.shape, y.shape)
    print(dataset)

    #
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(X.permute(1, 2, 0).numpy(), cmap = "gray")
    # plt.imshow(y.permute(1, 2, 0).numpy(), cmap ="gray")
    # plt.show()

    dataset = DeepLenseDiffusionDataset(diffusion_helper=DiffusionHelper())

    X = dataset[0]

    print(X.shape)
    print(dataset)

    from matplotlib import pyplot as plt

    plt.imshow(X.permute(1, 2, 0).numpy(), cmap = "gray")
    plt.show()
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

__all__ = [
    "ParticleJetDataset",
]

class ParticleJetDataset(Dataset):
    def __init__(self, path=None):
        super().__init__()

        if path is None:
            path = "../../data/quark_gluon_genie_falcon_dataset_n139306.hdf5"

        self.path = path

        self.files = h5py.File(path, "r")

        self.data = self.files.get("X_jets")
        self.targets = self.files.get("y")

    def __getitem__(self, index):
        # jet_img = torch.tensor(np.asarray(self.data[index], dtype=np.float32))
        jet_img = torch.tensor(self.data[index][:], dtype=torch.float32)

        if jet_img.ndim > 3:
            new_dims = (0, 3, 1, 2)
        else:
            new_dims = (2, 0, 1)
        return (
            jet_img.permute(*new_dims),
            torch.tensor(np.asarray(self.targets[index])).int()
        )

    def __getitems__(self, indices):
        return (
            torch.as_tensor(np.asarray(self.data[indices], dtype=np.float32)).permute(0, 3, 1, 2),
            torch.as_tensor(np.asarray(self.targets[indices])).int()
        )

    def to_image(self, index):
        img, _ = self.__getitem__(index)
        img = img.permute(1, 2, 0).cpu().numpy()
        print(img)
        print(img.dtype)
        print(img.shape)
        # return Image.fromarray(img.astype(np.int64))
        return img

if __name__ == "__main__":
    dataset = ParticleJetDataset()
    #
    # A, B = dataset.__getitem__([0, 1, 2, 3, 4])
    #
    # print(A.shape, B.shape)
    # print(A.dtype, B.dtype)
    index = 10

    img = dataset.to_image(index=index)

    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()

# -*- coding: utf-8 -*-

import torch
from torch import nn

from matplotlib import pyplot as plt

__all__ = [
    "LipchitzCriterion",
    "MSELipchitzCriterion",
    "_convolution_output_size",
    "generate_tuple",
    "generate_spatial_size",
    "convolution_output_size",
]

class LipchitzCriterion(nn.Module):
    def __init__(self, reduction="sum"):

        super().__init__()

        assert reduction in ['sum', 'mean', 'none'], "Reduction must be 'sum', 'mean', or 'none'."

        self.reduction = reduction

    def forward(self, input, target):
        reduce_dim = list(range(1, len(input.shape)))
        if self.reduction == 'sum':
            input = torch.sum(input, dim=reduce_dim)
            target = torch.sum(target, dim=reduce_dim)

        elif self.reduction == 'mean':
            input = torch.mean(input, dim=reduce_dim)
            target = torch.mean(target, dim=reduce_dim)
        else:
            pass

        input_forward = torch.diff(input, dim=0, append=input[0].unsqueeze(0))
        target_forward = torch.diff(target, dim=0, append=target[0].unsqueeze(0))

        indx = torch.randperm(len(input))

        input_backward = torch.diff(input[indx], dim=0, append=input[indx][-1].unsqueeze(0))
        target_backward = torch.diff(target[indx], dim=0, append=target[indx][-1].unsqueeze(0))

        # input_forward = input_forward[idx]
        # target_forward = target_forward[idx]

        input = input_forward + input_backward
        target = target_forward + target_backward

        return torch.pow((input_forward - input_backward) - (target_forward - target_backward), 2).mean()

class MSELipchitzCriterion(nn.Module):
    def __init__(self, lipchitz_factor=0., lipchitz_reduction="mean", mse_reduction="mean"):
        super().__init__()

        assert mse_reduction in ['sum', 'mean', 'none'], "MSE Reduction must be 'sum', 'mean', or 'none'."
        assert lipchitz_reduction in ['sum', 'mean', 'none'], "Lipchitz Reduction must be 'sum', 'mean', or 'none'."

        assert 0 <= lipchitz_factor <= 1, "Lipchitz factor must be float between 0 and 1."

        self.lipchitz_factor = lipchitz_factor
        self.lipchitz_reduction = lipchitz_reduction
        self.mse_reduction = mse_reduction

        self.mse_criterion = nn.MSELoss(reduction=mse_reduction)
        self.lipchitz_criterion = LipchitzCriterion(reduction=lipchitz_reduction)

    def forward(self, input, target):
        mse_loss = (1 - self.lipchitz_factor) * self.mse_criterion(input, target)
        lipchitz_loss = self.lipchitz_factor * self.lipchitz_criterion(input, target)

        return mse_loss + lipchitz_loss

def _convolution_output_size(input_size, kernel_size, stride, padding, dilation=1):
    return int((input_size - (kernel_size + (kernel_size - 1) * (dilation - 1)) + stride + (2 * padding)) / stride)

def generate_tuple(parameter):
    if not isinstance(parameter, (list, tuple)):
        parameter = (parameter, parameter)
    return parameter

def convolution_output_size(input_size, kernel_size, stride, padding, dilation=1):
    input_size = generate_tuple(input_size)
    kernel_size = generate_tuple(kernel_size)
    stride = generate_tuple(stride)
    padding = generate_tuple(padding)
    dilation = generate_tuple(dilation)

    return (
        _convolution_output_size(
            input_size=input_size[0],
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0]
        ),
        _convolution_output_size(
            input_size=input_size[1],
            kernel_size=kernel_size[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
        ),
    )

def generate_spatial_size(input_size, kernel_size, stride, padding, dilation, channels):
    sizes = []
    bias = 0
    for _ in range(len(channels)):
        size_ = []

        size_.append(input_size if isinstance(input_size, int) else input_size[0])
        input_size = convolution_output_size(
            input_size=input_size,
            kernel_size=kernel_size + bias,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        bias += 3

        size_.append(input_size[0])

        sizes.append(size_)

    return sizes[-1][-1]

@torch.no_grad()
def visualize_images(batch, folder="mnist", nrows=6, iteration = None, figsize=(10, 15)):
    imgs = torch.split(batch.permute(0, 2, 3, 1), 1, dim = 0)
    imgs = [img.numpy().squeeze(0) for img in imgs]

    ncols = len(batch) // nrows
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=figsize)

    for row in range(nrows):
        for col in range(ncols):
            index = row * ncols + col
            ax[row, col].imshow(imgs[index] * .5 + .5, cmap="gray")
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # print(index)

    if iteration is not None:
        title = f"Visualized image scans: iteration {iteration}"
    else:
        title = "Visualized image scans"

    plt.title(title)

    plt.show()
    plt.savefig(f"images/{folder}/iteration_{iteration}.png")
    plt.close("all")

    return
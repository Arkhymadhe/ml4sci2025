# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torch.nn import functional as F

from typing import List, Union, Sequence

__all__ = [
    "VGGContentLoss",
    "FourierTransformLoss",
    "SuperResolutionLoss"
]

class VGGContentLoss(nn.Module):
    reducers = {
        "mean": torch.mean,
        "sum": torch.sum,
        "none": lambda x: x,
    }

    def __init__(
        self,
        layers: Union[int, List[int]] = 8,
        layer_weights: Union[int, List[int], torch.Tensor, None] = None,
        progressive_weights: bool = False,
        reduction: str = "mean",
    ):
        super(VGGContentLoss, self).__init__()

        assert reduction in ["mean", "sum", "none"], f"reduction must be one of {['mean', 'sum', 'none']}"

        self.reducer_function = self.reducers[reduction]

        self.layers = layers if isinstance(layers, list) else [layers]

        if layer_weights is None:
            if progressive_weights:
                layer_weights = [
                    (i + 1) / len(self.layers)
                    for i in range(len(self.layers))
                ]
                layer_weights = torch.tensor(layer_weights, dtype=torch.float)
            else:
                layer_weights = torch.ones(len(self.layers)) / len(self.layers)

        self.layer_weights = layer_weights

        self.vgg_model = torchvision.models.vgg19(pretrained=True).features

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = len(input)
        loss = torch.zeros_like(self.layer_weights)
        print("Input shape: ", input.shape)

        for i, layer_idx in enumerate(self.layers):
            x = torch.cat((input, target), dim=0)
            y = self.vgg_model[:layer_idx+1](x)

            input_, target_ = torch.split(y, split_size_or_sections=batch_size, dim=0)

            layer_loss = F.mse_loss(input_, target_)
            loss[i] = layer_loss

        print("Input shape: ", input.shape)
        print("Target shape: ", target.shape)

        loss = loss * self.layer_weights

        return self.reducer_function(loss)

class FourierTransformLoss(nn.Module):
    def __init__(self, amplitude_weight = .5):
        super().__init__()

        self.amplitude_weight = amplitude_weight

    @staticmethod
    def fourier_transform(x: torch.Tensor) -> Sequence[torch.Tensor]:
        x_fft = torch.fft.fftn(x)
        x_fft = torch.fft.fftshift(x_fft)

        real_x_fft, imag_x_fft = x_fft.real, x_fft.imag

        amplitude = torch.pow(real_x_fft**2 + imag_x_fft**2, .5)
        phase = torch.atan2(imag_x_fft, real_x_fft)

        return amplitude, phase

    @staticmethod
    def inverse_fourier_transform(
        amplitude: torch.Tensor,
        phase: torch.Tensor
    ) -> torch.Tensor:
        imag = amplitude * torch.sin(phase)
        real = amplitude * torch.cos(phase)

        complex_tensor = torch.complex(real, imag)
        complex_tensor = torch.fft.ifftshift(complex_tensor)
        complex_tensor = torch.fft.ifftn(complex_tensor)

        return complex_tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_amplitude, input_phase = self.fourier_transform(input)
        target_amplitude, target_phase = self.fourier_transform(target)

        amplitude_loss = F.l1_loss(input_amplitude, target_amplitude)
        phase_loss = F.l1_loss(input_phase, target_phase)

        return self.amplitude_weight * amplitude_loss + (1 - self.amplitude_weight) * phase_loss

class SuperResolutionLoss(nn.Module):
    def __init__(
        self,
        fourier_loss_weight: float = .5,
        amplitude_weight: float = .5,
        layers: Union[int, List[int]] = 8,
        layer_weights: Union[int, List[int], torch.Tensor, None] = None,
        progressive_weights: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.fourier_loss_weight = fourier_loss_weight

        self.fourier_criterion = FourierTransformLoss(amplitude_weight=amplitude_weight)
        self.vgg_criterion = VGGContentLoss(
            layers=layers,
            layer_weights=layer_weights,
            progressive_weights=progressive_weights,
            reduction=reduction
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fourier_loss = self.fourier_criterion(input, target)
        vgg_loss = self.vgg_criterion(input, target)

        print("fourier loss: ", fourier_loss)
        print("vgg loss: ", vgg_loss)
        return self.fourier_loss_weight * fourier_loss + (1 - self.fourier_loss_weight) * vgg_loss

def calculate_fft(x):
    # x = x.flatten()
    fft_im = torch.fft.fftshift(torch.fft.fftn(x.clone()))  # bx3xhxw
    fft_amp = fft_im.real**2 + fft_im.imag**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im.imag, fft_im.real)
    return fft_amp, fft_pha

def calculate_inverse_fft(fft_amp, fft_pha):
    imag = fft_amp * torch.sin(fft_pha)
    real = fft_amp * torch.cos(fft_pha)
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifftn(torch.fft.ifftshift(fft_y))
    return y.real

if __name__ == '__main__':
    img = plt.imread("full-keilah-kang.jpg")
    x = torch.randn(1, 3, 224, 224)
    x_ = x + .001 * torch.randn(1, 3, 224, 224)
    #
    # criterion = SuperResolutionLoss(layers = [8, 17])
    #
    # image = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 225
    # image_tensor = torch.cat([image.permute(0, 3, 1, 2) for _ in range(1)], dim=0)
    #
    # print("IMG SHAPE:", image.shape)
    #
    # loss = criterion(image_tensor, image_tensor)
    # print(loss)

    image = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 225

    amp, ph = calculate_fft(image)
    new_image = calculate_inverse_fft(amp, ph)

    print(amp.shape)
    print(ph.shape)
    print(image.flatten().shape)

    plt.imshow(ph.squeeze().numpy())
    plt.show()

    plt.imshow(amp.squeeze().numpy())
    plt.show()

    image_ = new_image.view(image.shape).squeeze().numpy()

    plt.imshow(image_)
    plt.show()

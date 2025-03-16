# -*- coding: utf-8 -*-

from torch import nn
from src.utils import generate_spatial_size

__all__ = [
    "Encoder",
    "Decoder",
    "SpatialTransformer",
    "AutoEncoder",
]


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=1, padding=0, dilation=1, cd=3):
        super().__init__()

        self.channels = channels[:len(channels) // 2]

        self.kernel_size = kernel_size,
        self.stride = stride,
        self.padding = padding,
        self.dilation = dilation
        self.cd = cd

        layers = []
        bias = 0

        for i, channels_ in enumerate(channels):
            layer = nn.Conv2d(
                in_channels=channels_[0],
                out_channels=channels_[1],
                kernel_size=kernel_size + bias,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            bias += self.cd
            layers.append(layer)

            if i + 1 == len(self.channels):
                continue
            else:
                bn = nn.BatchNorm2d(channels_[1])
                act = nn.LeakyReLU(.2)

                layers.append(bn)
                layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpatialTransformer(nn.Module):
    def __init__(self, num_channels, spatial_size, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.spatial_size = spatial_size

        self.transform_layer = nn.Linear(
            in_features=int(num_channels * spatial_size ** 2),
            out_features=latent_dim
        )
        self.inverse_transform_layer = nn.Linear(
            in_features=latent_dim,
            out_features=int(num_channels * spatial_size ** 2),
        )

    def extract_latent_representation(self, x):
        return self.transform_layer(x.flatten(start_dim=1))

    def reconstruct_spatial_signal(self, x):
        return x.view(-1, self.num_channels, self.spatial_size, self.spatial_size)

    def forward(self, x):
        x = self.inverse_transform_layer(self.extract_latent_representation(x))
        return self.reconstruct_spatial_signal(x)


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=1, padding=0, dilation=1, cd=3):
        super().__init__()

        R = []

        channels_ = list(reversed(channels))

        for r in channels_:
            R.append(list(reversed(r)))
            # R.append(r)

        self.channels = R

        self.kernel_size = kernel_size,
        self.stride = stride,
        self.padding = padding,
        self.dilation = dilation
        self.cd = cd

        layers = []
        bias = 0

        for i, c_ in enumerate(self.channels):
            bias += self.cd

            layer = nn.ConvTranspose2d(
                in_channels=c_[0],
                out_channels=c_[1],
                kernel_size=kernel_size - bias,
                stride=stride,
                output_padding=padding,
                dilation=dilation
            )

            layers.append(layer)

            if i + 1 == len(self.channels):
                layers.append(nn.Tanh())
                continue
            else:
                bn = nn.BatchNorm2d(c_[1])
                act = nn.LeakyReLU(.2)

                layers.append(bn)
                layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        channels=None,
        input_size=64,
        latent_dim=64,
        kernel_size=2,
        stride=1,
        padding=0,
        dilation=1,
        cd = 3
    ):
        super().__init__()

        self.latent_dim = latent_dim

        if encoder is None:
            encoder = Encoder(
                channels = channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                cd=cd
            )

        if decoder is None:
            num_conv = len(list(filter(lambda x: "Conv" in x.__class__.__name__, encoder.layers)))
            decoder = Decoder(
                channels = channels,
                kernel_size=kernel_size + (cd * num_conv),
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.encoder = encoder
        self.decoder = decoder
        self.spatial_transformer = SpatialTransformer(
            num_channels=channels[-1][-1],
            spatial_size=generate_spatial_size(
                input_size=input_size,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            latent_dim=latent_dim
        )

    def forward(self, x):
        return self.decoder(self.spatial_transformer(self.encoder(x)))
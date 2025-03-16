# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchinfo import summary

from typing import Union, Optional

__all__ = [
    "MaskedAutoEncoder",
    "MaskTransformer",
    "FourierDiscriminator"
]

class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size = 128,
        hidden_size = 1024,
        num_heads=32,
        forward_dim=2048,
        num_encoder_layers=3
    ):
        super(MaskedAutoEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.num_encoder_layers = num_encoder_layers

        self.projector = nn.Linear(in_features=input_size, out_features = hidden_size)
        self.encoder = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            batch_first=True
        ).encoder

    def forward(self, x):
        x = self.projector(x)
        return self.encoder(x)

class MaskTransformer(nn.Module):
    def __init__(self, use_mask=False, mask_token = -100, mask_ratio=.75):
        super(MaskTransformer, self).__init__()

        self.mask_ratio = mask_ratio
        self.use_mask = use_mask
        self.mask_token = mask_token

    def mask_patches(self, patched_image):
        num_patches = patched_image.shape[1]
        num_batches = len(patched_image)

        num_patches_to_mask = int(num_patches * self.mask_ratio)
        num_patches_to_mask = 2

        patch_indices = torch.randint(
            low=0,
            high=num_patches,
            size=(num_batches, num_patches_to_mask)
        )

        print("Patch indices:", patch_indices.shape)
        print(patch_indices)

        patch_mask = torch.ones(num_batches, num_patches).bool()
        patch_mask[:, patch_indices] = False

        print("Patch mask:", patch_mask.shape)
        print(patch_mask)

        if self.use_mask:
            batch_index, patch_index = torch.where(~patch_mask)
            patched_image[batch_index, patch_index, :] = self.mask_token
        else:
            batch_index, patch_index = torch.where(patch_mask)
            patched_image = patched_image[batch_index, patch_index, :]

        return patched_image, patch_mask

    @torch.inference_mode()
    def forward(self, x):
        return self.mask_patches(x)

class FourierDiscriminator(nn.Module):
    def __init__(self, image_size=224):
        super(FourierDiscriminator, self).__init__()

        model_dim = 1024

        self.model = nn.Sequential(
            nn.Linear(in_features=image_size**2 * 3, out_features=model_dim),
            nn.GELU(),
            nn.Linear(in_features=model_dim, out_features=model_dim),
            nn.GELU(),
            nn.Linear(in_features=model_dim, out_features=model_dim),
            nn.GELU(),
            nn.Linear(in_features=model_dim, out_features=model_dim),
            nn.GELU(),
            nn.Linear(in_features=model_dim, out_features=1),
        )

    def forward(self, x):
        x = x.flatten()
        return self.model(x)

class DiffusionHelper:
    def __init__(
        self,
        max_beta = .02,
        min_beta = .0001,
        num_time_steps = 1000
    ):
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.num_time_steps = num_time_steps

        self.betas = torch.linspace(self.min_beta, self.max_beta, self.num_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def corrupt_sample(
        self,
        x: Union[int, torch.Tensor],
        t: Union[int, torch.Tensor] = 0
    ):
        eps = torch.randn_like(x).clamp(0, 1)
        x = torch.pow(self.alpha_bar[t], .5) * x + torch.pow(1 - self.alpha_bar[t], .5) * eps
        return x, eps

    def restore_sample(
        self,
        x: torch.Tensor,
        eps: Optional[Union[int, torch.Tensor]] = None,
        t: Union[int, torch.Tensor] = 0
    ):
        if eps is None:
            eps = torch.randn_like(x).clamp(0, 1)
        return (x - torch.pow(1 - self.alpha_bar[t], .5) * eps) / torch.pow(self.alpha_bar[t], .5)

    def sample_from_time_steps(self, num_time_steps: int =32):
        time_steps = torch.randint(low=0, high = self.num_time_steps, size=(num_time_steps,))
        return time_steps

    def generate_corrupted_samples(
        self,
        x: Union[int, torch.Tensor],
        num_time_steps: int = 1
    ):
        t = self.sample_from_time_steps(num_time_steps)
        return self.corrupt_sample(x=x, t=t), t


class UNetConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size = 3,
        in_channels = 3,
        out_channels = 3,
        stride = 1,
        padding = 1,
        dilation = 1,
        pool = False,
    ):
        super().__init__()

        self.pool = pool
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation
        )
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return F.max_pool2d(x, kernel_size=2) if self.pool else x


class UNetConvTransposeBlock(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        in_channels=3,
        out_channels=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels*2,
            out_channels=in_channels,
            kernel_size=2,
            stride = 1,
            dilation = 2,
            padding = 1,
        )

        self.conv_block = UNetConvBlock(
            kernel_size = kernel_size,
            in_channels = in_channels*2,
            out_channels = in_channels,
            stride = stride,
            padding = padding,
            dilation = dilation,
            pool = False,
        )

    def forward(self, x, residual_x):
        x = F.upsample(x, scale_factor=2)
        print("x (before  conv): ", x.shape)
        x = self.conv1(x)

        print("Residual x: ", residual_x.shape)
        print("x (after conv): ", x.shape)

        x = torch.cat([x, residual_x], dim=1)

        return self.conv_block(x)

class UNetEncoder(nn.Module):
    def __init__(
        self,
        num_blocks = 3,
        kernel_size=3,
        in_channels=3,
        out_channels=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        channels = [
            ((in_channels if i == 0 else out_channels * 2 ** (i - 1)), out_channels * 2 ** i)
            for i in range(0, num_blocks,)
        ]

        print("Encoder channels: ", channels)

        blocks = [
            UNetConvBlock(
                kernel_size=kernel_size,
                in_channels=channel_pair[0],
                out_channels=channel_pair[1],
                stride=stride,
                padding=padding,
                dilation=dilation,
                pool=i != len(channels),
            )
            for i, channel_pair in enumerate(channels, 1)
        ]

        self.blocks = nn.Sequential(*blocks)

        self.activation_bank = []

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            self.activation_bank.append(x)
        return x

class UNetBottleneck(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        in_channels=3,
        out_channels=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.layer = UNetConvBlock(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pool=True # TODO: Bears investigating; should be False ideally.
        )

    def forward(self, x):
        return self.layer(x)

class UNetDecoder(nn.Module):
    def __init__(
        self,
        num_blocks=3,
        kernel_size=3,
        in_channels=3,
        out_channels=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        channels = [
            (in_channels * 2 ** i, (out_channels if i == 0 else in_channels * 2 ** (i - 1)))
            for i in reversed(range(0, num_blocks))
        ]

        print("Decoder channels: ", channels)

        self.blocks = [
            UNetConvTransposeBlock(
                kernel_size=kernel_size,
                in_channels=channel_pair[0],
                out_channels=channel_pair[1],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for i, channel_pair in enumerate(channels)
        ]

        self.blocks = nn.Sequential(*self.blocks)

        self.activation_bank = []

    def forward(self, x, residual_xs):
        num_blocks = len(residual_xs)
        for i, block in enumerate(self.blocks):
            residual_x = residual_xs[num_blocks - i - 1]
            x = block(x, residual_x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        num_blocks=3,
        kernel_size=3,
        in_channels=3,
        out_channels=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.num_blocks = num_blocks

        self.encoder = UNetEncoder(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_blocks=num_blocks,
        )

        self.bottleneck = UNetBottleneck(
            kernel_size=kernel_size,
            in_channels=out_channels * 2 ** (num_blocks - 1),
            out_channels=out_channels * 2 ** num_blocks,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.decoder = UNetDecoder(
            kernel_size=kernel_size,
            in_channels=out_channels,
            out_channels=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_blocks=num_blocks,
        )

    def forward(self, x):
        x_ = self.encoder(x)
        residual_xs = self.encoder.activation_bank
        x_ = self.bottleneck(x_)
        return self.decoder(x_, residual_xs)

if  __name__ == "__main__":
    hidden_size = 1024

    # batch_size = 3
    # num_patches = 10
    # num_features = 5
    #
    # x = torch.randn(batch_size, num_patches, num_features)
    #
    # model = MaskTransformer(use_mask=True)
    #
    # y1, y2 = model(x)
    #
    # print(x.shape)
    # print("y1 shape: ", y1.shape)
    # print(y1)
    # print("y2 shape: ", y2.shape)
    # print(y2)

    ### Test diffusion helper

    import matplotlib.pyplot as plt

    # img = plt.imread("full-keilah-kang.jpg")
    #
    # plt.imshow(img)
    # plt.title("Original image")
    # plt.show()
    #
    # img_tensor = torch.tensor(img, dtype = torch.float32).unsqueeze_(0).permute(0, 3, 1, 2) / 255.
    # batch_img_tensor = torch.cat([img_tensor for _ in range(2)], dim=0)
    # diffusion_helper = DiffusionHelper()
    #
    # t = 500
    # # img_corrupted_tensor, noise = diffusion_helper.corrupt_sample(batch_img_tensor, t = t)
    # output = diffusion_helper.generate_corrupted_samples(batch_img_tensor)
    # img_corrupted_tensor, noise, t_ = output[0][0], output[0][1], output[1]
    #
    # img_corrupted = img_corrupted_tensor[0].squeeze().permute(1, 2, 0).numpy()
    #
    # plt.imshow(img_corrupted)
    # plt.title("Corrupted image, t = {}".format(t))
    # plt.show()
    #
    # img_restored_tensor = diffusion_helper.restore_sample(img_corrupted_tensor, eps=noise, t=t_)[0]
    #
    # img_restored = img_restored_tensor.squeeze().permute(1, 2, 0).numpy()
    #
    # plt.imshow(img_restored)
    # plt.title("Restored corrupted image, t = {}".format(t))
    # plt.show()

    in_channels = 3
    out_channels = 64
    num_blocks = 4

    # block1 = UNetConvBlock(in_channels=in_channels, out_channels=out_channels, pool=True)
    # block2 = UNetConvTransposeBlock(in_channels=out_channels, out_channels=in_channels)
    # block = nn.Sequential(block1, block2)
    #
    # block1 = UNetEncoder(in_channels=in_channels, out_channels=out_channels, num_blocks=num_blocks)
    # block2 = UNetDecoder(in_channels=out_channels, out_channels=in_channels, num_blocks=num_blocks)
    # bneck = UNetBottleneck(in_channels=out_channels * 2**(num_blocks-1), out_channels=out_channels* 2**num_blocks)
    #
    # block = nn.Sequential(block1, bneck, block2)

    unet = UNet(in_channels=in_channels, out_channels=out_channels, num_blocks=num_blocks)

    x = torch.randn(1, in_channels, 160, 160)

    print(unet)

    print(summary(unet, input_data=x))

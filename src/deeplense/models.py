# -*- coding: utf-8 -*-

import gc
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import Dataset, DataLoader

from torch.nn import functional as F
from torchinfo import summary

from typing import Union, Optional, Sequence

__all__ = [
    "MaskedAutoEncoder",
    "MaskTransformer",
    "FourierDiscriminator",
    "DiffusionHelper",
    "UNetConvBlock",
    "UNetConvTransposeBlock",
    "UNetEncoder",
    "UNetDecoder",
    "UNetBottleneck",
    "UNetSelfAttention",
    "UNetSelfAttentionBottleneck",
    "UNet",
    "UNetTransformer",
]


class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size=128,
        hidden_size=1024,
        num_heads=32,
        forward_dim=2048,
        num_encoder_layers=3,
    ):
        super(MaskedAutoEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.num_encoder_layers = num_encoder_layers

        self.projector = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.encoder = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            batch_first=True,
        ).encoder

    def forward(self, x):
        x = self.projector(x)
        return self.encoder(x)


class MaskTransformer(nn.Module):
    def __init__(self, use_mask=False, mask_token=-100, mask_ratio=0.75):
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
            low=0, high=num_patches, size=(num_batches, num_patches_to_mask)
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
        self: "DiffusionHelper",
        max_beta: float = 2e-2,
        min_beta: float = 1e-4,
        num_time_steps: int = 1000,
        device: str = "cpu"
    ):
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.num_time_steps = num_time_steps
        self.device = str(device)

        self.betas = torch.linspace(self.min_beta, self.max_beta, self.num_time_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def corrupt_sample(
        self: "DiffusionHelper",
        x: torch.Tensor,
        t: Union[int, torch.Tensor] = 0,
    ) -> Sequence[torch.Tensor]:
        batch = True
        if x.ndim == 3:
            x = x.unsqueeze(0)
            batch = False
            if isinstance(t, int):
                t = torch.tensor([t])

        x, eps = self.corrupt_batch(x=x.to(self.device), t=t.to(self.device))

        return (x, eps) if batch else (x.squeeze(0), eps.squeeze(0))

    def corrupt_batch(
        self: "DiffusionHelper",
        x: Union[int, torch.Tensor],
        t: Union[int, torch.Tensor] = 0,
    ) -> Sequence[torch.Tensor]:
        eps = torch.randn_like(x).clamp(0, 1)
        alpha_bar = self.alpha_bar[t][:, None, None, None]

        x = (
            torch.pow(alpha_bar, 0.5) * x
            + torch.pow(1 - alpha_bar, 0.5) * eps
        )
        return x, eps

    def restore_sample_(
        self: "DiffusionHelper",
        x_t: torch.Tensor,
        noise: Optional[Union[int, torch.Tensor]] = None,
        t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_t).clamp(0, 1)

        eps = torch.randn_like(x_t).clamp(0, 1)
        # pred = x - torch.pow(1 - self.alpha_bar[t], 0.5) * noise

        noise = (1 - self.alphas[t]) * noise / torch.pow(1 - self.alpha_bar[t], 0.5)
        noise = (x_t - noise) / torch.pow(self.alphas[t], 0.5)

        return noise + eps * torch.pow(self.betas[t], .5)

    def restore_sample(
        self: "DiffusionHelper",
        x_t: torch.Tensor,
        noise: Optional[Union[int, torch.Tensor]] = None,
        t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        batch = True
        if x_t.ndim == 3:
            batch = False
            x_t = x_t.unsqueeze(0)

        x = self.restore_batch(x_t=x_t, noise_hat=noise, t=t)

        return x if batch else x.squeeze(0)

    def restore_batch(
        self: "DiffusionHelper",
        x_t: torch.Tensor,
        noise_hat: Optional[Union[int, torch.Tensor]],
        t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        alphas = self.alphas[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        betas = 1 - alphas

        z = torch.randn_like(x_t)
        z[t == 1] = 0

        noise = betas * noise_hat / torch.sqrt(1 - alpha_bar)
        noise = (x_t - noise) / torch.sqrt(alphas)

        return noise + z * torch.sqrt(betas)

    def sample_time_steps(
        self: "DiffusionHelper", num_time_steps: int = 32
    ) -> torch.Tensor:
        time_steps = torch.randint(
            low=1, high=self.num_time_steps, size=(num_time_steps,)
        )
        return time_steps

    def generate_image_samples(self, model, x_t=None, batch_size = 1, channels = 1, input_size=None):
        model = model.eval().to(self.device)
        if x_t is None:
            x_t = torch.randn(batch_size, channels, input_size, input_size)

        x_t = x_t.to(self.device)

        for t_ in reversed(range(1, self.num_time_steps)):
            t = torch.ones(batch_size, dtype = torch.int32).fill_(t_).to(self.device)
            noise_hat = model(x_t, t)
            x_t = self.restore_batch(x_t=x_t, t=t, noise_hat=noise_hat)
            print(f"Time step: {t_}")

            gc.collect()

        return x_t

    def generate_corrupted_samples(
        self: "DiffusionHelper", x: torch.Tensor, num_time_steps: int = 1
    ) -> Sequence[torch.Tensor]:
        t = self.sample_time_steps(num_time_steps)

        output = self.corrupt_sample(x=x, t=t)
        corrupted_samples, noise = output

        return corrupted_samples, noise, t

    def __str__(self):
        return f"{self.__class__.__name__}(min_beta={self.min_beta}, max_beta={self.max_beta}, num_time_steps={self.num_time_steps}, device={self.device})"

    def __repr__(self):
        return self.__str__()


class UNetConvBlock(nn.Module):
    def __init__(
        self: "UNetConvBlock",
        embedding_dim: int = 512,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        pool: bool = False,
    ):
        super().__init__()

        self.pool = pool
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.time_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=embedding_dim, out_features=out_channels)
        )

    def forward(self: "UNetConvBlock", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        if self.pool:
            x = F.max_pool2d(x, kernel_size=2)

        xt = self.time_projector(t)[:, :, None, None]
        xt = xt.repeat(1, 1, *x.shape[-2:])

        return x + xt


class UNetConvTransposeBlock(nn.Module):
    def __init__(
        self: "UNetConvTransposeBlock",
        kernel_size: int = 3,
        embedding_dim: int = 512,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=2,
            stride=1,
            dilation=2,
            padding=1,
        )

        self.conv_block = UNetConvBlock(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pool=False,
        )

        self.time_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=embedding_dim, out_features=in_channels)
        )

    def forward(
        self: "UNetConvTransposeBlock", x: torch.Tensor, t: torch.Tensor, residual_x: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)

        x = torch.cat(tensors=[x, residual_x], dim=1)
        x = self.conv_block(x, t)

        xt = self.time_projector(t)[:, :, None, None]
        xt = xt.repeat(1, 1, *x.shape[-2:])

        return x + xt


class UNetEncoder(nn.Module):
    def __init__(
        self: "UNetEncoder",
        num_blocks: int = 3,
        embedding_dim: int = 512,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        channels = [
            (
                (in_channels if i == 0 else out_channels * 2 ** (i - 1)),
                out_channels * 2**i,
            )
            for i in range(
                0,
                num_blocks,
            )
        ]

        blocks = [
            UNetConvBlock(
                kernel_size=kernel_size,
                embedding_dim=embedding_dim,
                in_channels=channel_pair[0],
                out_channels=channel_pair[1],
                stride=stride,
                padding=padding,
                dilation=dilation,
                pool=True,
                # pool=i != len(channels),
            )
            for i, channel_pair in enumerate(channels, 1)
        ]

        self.blocks = nn.Sequential(*blocks)

        self.activation_bank = []

    def forward(self: "UNetEncoder", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, t)
            self.activation_bank.append(x)
        return x


class UNetBottleneck(nn.Module):
    def __init__(
        self: "UNetBottleneck",
        kernel_size: int = 3,
        embedding_dim: int = 512,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.layer = UNetConvBlock(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pool=True,  # TODO: Bears investigating; should be False ideally.
        )

    def forward(self: "UNetBottleneck", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.layer(x, t)


class UNetSelfAttentionBottleneck(nn.Module):
    def __init__(
        self: "UNetSelfAttentionBottleneck",
        in_channels: int = 3,
        feature_dim: int = 1024,
        embedding_dim: int = 512,
        num_heads: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=2,
            dilation=1,
            padding=0,
        )

        self.time_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=embedding_dim, out_features=in_channels)
        )

    def forward(self: "UNetSelfAttentionBottleneck", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # FIXME: Allow time information to pass through bottleneck. Forgot that before!
        xt = self.time_projector(t)[:, :, None, None]
        xt = xt.repeat(1, 1, *x.shape[-2:])

        x += xt

        B, C, H, W = x.shape
        x = x.view(B, C, H * W)

        values = F.scaled_dot_product_attention(key=x, query=x, value=x)

        return self.conv1((values + x).contiguous().view(B, C, H, W))


class UNetSelfAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels: Optional[int] = None, feature_dim=1024, num_heads=32, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )

        if out_channels is None:
            out_channels = in_channels * 2

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            dilation=1,
            padding=0,
        )

    def forward(self, key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        key = key.flatten(2)

        B, C, H, W = query.shape
        query = query.view(B, C, H, W)

        results = F.scaled_dot_product_attention(key=key, query=query, value=key, is_causal=False)

        return self.conv1(results.contiguous().view(B, C, H, W) + query.contiguous().view(B, C, H, W))


class UNetDecoder(nn.Module):
    def __init__(
        self: "UNetDecoder",
        num_blocks: int = 3,
        embedding_dim: int = 512,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        channels = [
            (
                in_channels * 2**i,
                (out_channels if i == 0 else in_channels * 2 ** (i - 1)),
            )
            for i in reversed(range(0, num_blocks))
        ]

        self.blocks = [
            UNetConvTransposeBlock(
                kernel_size=kernel_size,
                embedding_dim = embedding_dim,
                in_channels=channel_pair[0],
                out_channels=channel_pair[1],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for i, channel_pair in enumerate(channels)
        ]

        self.blocks = nn.Sequential(*self.blocks)
        self.final_conv = nn.ConvTranspose2d(
            in_channels=channels[-1][0],
            out_channels=channels[-1][-1],
            kernel_size=2,
            stride=2,
            dilation=1,
            padding=0,
        )

    def forward(
        self: "UNetDecoder", x: torch.Tensor, t: torch.Tensor, residual_xs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        num_blocks = len(residual_xs)
        for i, block in enumerate(self.blocks):
            residual_x = residual_xs[num_blocks - i - 1]
            x = block(x, t, residual_x)

        return self.final_conv(x)


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        num_blocks: int = 3,
        embedding_dim: int = 512,
        num_timesteps: int = 1000,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.num_blocks = num_blocks

        self.time_embed = TimeEmbedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_timesteps
        )

        self.encoder = UNetEncoder(
            embedding_dim=embedding_dim,
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
            out_channels=out_channels * 2**num_blocks,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.decoder = UNetDecoder(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            in_channels=out_channels,
            out_channels=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_blocks=num_blocks,
        )

    def forward(self: "UNet", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(t)
        x_ = self.encoder(x, t)

        residual_xs = self.encoder.activation_bank.copy()
        x_ = self.bottleneck(x_, t)

        # FIXME: Had to clear the activation bank! Was hugging CUDA memory!
        self.encoder.activation_bank.clear()
        gc.collect()

        return torch.tanh(self.decoder(x_, t, residual_xs))


class UNetTransformer(nn.Module):
    def __init__(
        self: "UNet",
        num_blocks: int = 3,
        embedding_dim : int = 512,
        num_timesteps: int = 1000,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim

        self.time_embed = TimeEmbedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_timesteps
        )

        self.encoder = UNetEncoder(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_blocks=num_blocks,
        )

        self.bottleneck = UNetSelfAttentionBottleneck(
            in_channels=out_channels * 2 ** (num_blocks - 1),
            feature_dim=100,
            embedding_dim=embedding_dim,
            num_heads=5,
            dropout=0.1,
        )

        self.decoder = UNetDecoder(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            in_channels=out_channels,
            out_channels=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_blocks=num_blocks,
        )

    def forward(self: "UNet", x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(t).squeeze(-1)

        x_ = self.encoder(x, t)
        residual_xs = self.encoder.activation_bank.copy()
        x_ = self.bottleneck(x_, t)

        # FIXME: Had to clear the activation bank! Was hugging CUDA memory!
        self.encoder.activation_bank.clear()
        gc.collect()

        return torch.tanh(self.decoder(x_, t, residual_xs))


class TimeEmbedding(nn.Module):
    def __init__(self : "TimeEmbedding", num_embeddings : int = 1000, embedding_dim : int = 1024, dropout_rate : float =0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.dropout_rate = dropout_rate

        position = torch.arange(num_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-torch.log(torch.tensor([10000.0])) / embedding_dim)
        )

        pos_encoding = torch.zeros(num_embeddings, embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

        self.dropout = nn.Dropout1d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.pos_encoding[:, x, :].squeeze(0)
        return self.dropout(encoding)


if __name__ == "__main__":
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
    #
    # t = 512
    # diffusion_helper = DiffusionHelper(num_time_steps=t)
    #
    # # img_corrupted_tensor, noise = diffusion_helper.corrupt_sample(batch_img_tensor, t = t)
    # output = diffusion_helper.generate_corrupted_samples(batch_img_tensor, num_time_steps = 1)
    # img_corrupted_tensor, noise, t_ = output
    #
    # print("Sampled time steps: {}".format(t_))
    #
    # print(img_corrupted_tensor.shape)
    # print(output[0].shape)
    #
    # img_corrupted = img_corrupted_tensor[0].squeeze().permute(1, 2, 0).numpy()
    #
    # plt.imshow(img_corrupted)
    # plt.title("Corrupted image, t = {}".format(t))
    # plt.show()
    #
    # img_restored_tensor = diffusion_helper.restore_batch(x_t=img_corrupted_tensor, noise_hat=noise, t=t_)[0]
    #
    # img_restored = img_restored_tensor.squeeze().permute(1, 2, 0).numpy()
    #
    # plt.imshow(img_restored)
    # plt.title("Restored corrupted image, t = {}".format(t))
    # plt.show()

    # Test out the UNetTransformer

    in_channels = 3
    out_channels = 64
    num_blocks = 4
    embedding_dim = 512
    num_time_steps = 100

    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        num_timesteps=num_time_steps,
        embedding_dim=embedding_dim
    )
    unet_transformer = UNetTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        num_timesteps=num_time_steps,
        embedding_dim=embedding_dim
    )

    batch_size = 5

    x = torch.randn(batch_size, in_channels, 160, 160)
    t = torch.randint(0, num_time_steps, (batch_size, ))

    print(unet)
    summary(unet, input_data=[x, t])
    summary(unet_transformer, input_data=[x, t])

    prop = (
        100
        * sum(n.numel() for n in unet_transformer.parameters())
        / sum(n.numel() for n in unet.parameters())
    )
    print(prop)

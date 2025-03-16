# -*- coding: utf-8 -*-

import torch
from torch import nn

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric import nn as pygnn

from src.falcon.utils import generate_spatial_size
from src.falcon.attention import MultiHeadAttention

__all__ = [
    "Encoder",
    "Decoder",
    "SpatialTransformer",
    "VariationalSpatialTransformer",
    "AutoEncoder",
    "AdaptLayerForGNN",
    "NonLocalGNN",
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

class VariationalSpatialTransformer(SpatialTransformer):
    def __init__(self, num_channels, spatial_size, latent_dim):
        super().__init__(num_channels, spatial_size, 2*latent_dim)

        self.latent_dim = latent_dim

        self.inverse_transform_layer = nn.Linear(
            in_features=latent_dim,
            out_features=int(num_channels * spatial_size ** 2),
        )

    def forward(self, x):
        x_ = self.extract_latent_representation(x)
        mean, log_var = torch.split(x_, self.latent_dim, dim=1)

        eps = torch.randn_like(mean)
        z = mean + eps * log_var

        upsampled_z = self.inverse_transform_layer(z)

        return self.reconstruct_spatial_signal(upsampled_z)


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
        cd = 3,
        variational=False
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.variational = variational

        autoencoder_cls = VariationalSpatialTransformer if variational else SpatialTransformer

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
        self.spatial_transformer = autoencoder_cls(
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

class AdaptLayerForGNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, edge_index=None):
        return self.layers(x)


class NonLocalGNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_mlp_layers = 2,
        num_heads = 2,
        kernel_size=3,
        stride=1,
        mlp = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_mlp_layers = num_mlp_layers
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride

        # self.calibration_vector = torch.distributions.Normal(scale=1, loc=0).sample((1, out_features))
        self.calibration_vector = torch.linspace(0, 9, out_features).view(1, -1)
        self.calibration_vector.requires_grad_(True)

        if mlp:
            last_layer = nn.Linear(in_features, out_features)
            if num_mlp_layers > 1:
                layers = []
                for _ in range(num_mlp_layers-1):
                    layers.append(nn.Linear(in_features, in_features))

                layers.append(last_layer)

                last_layer = nn.Sequential(*layers)

            self.mlp = AdaptLayerForGNN(last_layer)

        else:
            self.mlp = pygnn.GCN(
                in_channels=in_features,
                hidden_channels=in_features,
                out_channels=out_features,
                num_layers=num_mlp_layers
            )

        self.attention = MultiHeadAttention(
            hidden_dim=out_features,
            state_dim=out_features,
            num_heads = num_heads,
            narrow=True
        )

        self.aggregator = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride
        )

    def local_embeddings(self, x, edge_index=None):
        return self.mlp(x=x, edge_index=edge_index)

    def calculate_attention(self, x):
        _ = self.attention(
            key=x,
            value=x,
            query=self.calibration_vector.expand(size = (x.shape[0], 1, -1))
        )
        return self.attention.attention_scores.mean(dim=1)

    def sort_graph_nodes_by_attention(self, x):
        attention_scores = self.calculate_attention(x)
        print("Attention scores shape: ", attention_scores.shape)
        print("Attention: ")
        print(attention_scores[0])
        new_indices = torch.argsort(attention_scores.squeeze(), dim=1, descending=False)
        print("New indices shape: ", new_indices.shape)
        return x[new_indices]

    def forward(self, x, edge_index=None):
        embeddings = self.local_embeddings(x, edge_index)
        print("1. Local embedding shape: ", embeddings.shape)

        sorted_x = self.sort_graph_nodes_by_attention(embeddings)
        print("2. Sorted embeddings shape: ", sorted_x.shape)

        aggregated_x = self.aggregator(torch.flatten(sorted_x, start_dim=1)).view(embeddings.shape)
        print("3. Non-locally aggregated nodes shape: ", aggregated_x.shape)

        final_x = torch.cat(tensors=[embeddings, aggregated_x], dim=1)
        print("4. Final concatenated shape: ", final_x.shape)
        return final_x

if __name__ == "__main__":
    num_batches = 32
    in_channels = 3
    out_channels = 128
    num_heads = 8

    input_size = 64
    kernel_size = 3
    latent_dim = 64
    stride = 1
    padding = 0
    dilation = 1

    num_nodes = 10
    mlp = True

    variational = False

    # channels = [
    #     [in_channels, 8],
    #     [8, 16],
    #     [16, 32],
    #     [32, 64],
    #     [64, 128],
    #     [128, 128]
    # ]
    #
    # model = AutoEncoder(
    #     latent_dim=latent_dim,
    #     variational=variational,
    #     channels=channels,
    #     input_size=input_size,
    #     kernel_size=kernel_size,
    #     dilation=dilation,
    #     stride=stride,
    #     padding=padding,
    # )
    #
    # x = torch.randn(10, in_channels, input_size, input_size)
    #
    # print(model(x).shape)
    graph_x = torch.randn(num_batches, num_nodes, in_channels)
    graph_x = torch.linspace(0, 99, (num_batches * num_nodes * in_channels)).view(num_batches, num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    edge_index = torch.arange(0, 2*num_nodes).view((2, num_nodes))
    edge_index[-1] = torch.tensor(list(reversed([_ for _ in range(num_nodes)])))


    gnn = NonLocalGNN(
        in_features=in_channels,
        out_features=out_channels,
        num_heads=num_heads,
        mlp=mlp
    )
    for param in gnn.mlp.parameters():
        param.requires_grad_(False)
        param.fill_(1.)

    print(gnn(graph_x, edge_index=edge_index).shape)

# -*- coding: utf-8 -*-

import torch
from torch import nn

__all__ = [
    "Attention",
    "MultiHeadAttention",
]


class Attention(nn.Module):
    def __init__(
        self,
        dropout_probability=0.2,
    ):
        super().__init__()

        self.attention_scores = None
        self.dropout_probability = dropout_probability

        self.dropout = nn.Dropout(p=self.dropout_probability)

    @classmethod
    def get_alignment_vectors(cls, key, query):
        alignment_vectors = torch.matmul(query, key.transpose(-2, -1))

        return alignment_vectors

    def forward(self, key, value, query=None, mask=None):
        if query is None:
            query = key.clone()

        alignment_vectors = self.get_alignment_vectors(key, query)

        if mask is not None:
            alignment_vectors.masked_fill_(~mask.unsqueeze(0).bool(), -torch.inf)

        alignment_vectors /= key.shape[-1] ** 0.5
        attention_scores = torch.softmax(alignment_vectors, dim=-1)
        attention_scores = self.dropout(attention_scores)

        self.attention_scores = attention_scores.clone().detach()

        return torch.matmul(attention_scores, value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        narrow=True,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        use_bias=False,
        dropout_probability=0.2,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self._attention_scores = None
        self.dropout_probability = dropout_probability

        self.narrow = narrow
        self.use_bias = use_bias

        self.key_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim, bias=use_bias
        )
        self.value_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim, bias=use_bias
        )
        self.query_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim, bias=use_bias
        )

        self.attention_heads = [
            Attention(dropout_probability=dropout_probability)
            for _ in range(1 if self.narrow else num_heads)
        ]

        print("Number of attentio heads: ", len(self.attention_heads))

        self.attention_heads = nn.ModuleList(self.attention_heads)

        if not narrow:
            self.context_transform = nn.Linear(
                in_features=sum([hidden_dim for _ in range(num_heads)]),
                out_features=self.hidden_dim,
                bias=True,
            )

        else:
            # self.query_transform = nn.LazyLinear(out_features=int(hidden_dim * 0.5))

            self.context_transform = nn.Linear(
                in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True
            )

    @property
    def attention_scores(self):
        return self._attention_scores

    @attention_scores.setter
    def attention_scores(self, new_attention_scores):
        self._attention_scores = new_attention_scores
        return

    def get_dims_per_head(self, x):
        N, L, H = x.shape
        x = x.view(N, L, self.num_heads, int(H / self.num_heads))
        print("New shape: ", N, L, self.num_heads, int(H / self.num_heads))
        return x.transpose(1, 2)

    def forward(self, key, value=None, query=None, mask=None):
        # query = self.query_transform(query) if self.narrow else query
        # context_vectors = [head(query, mask=mask) for head in self.attention_heads]

        if query is None:
            query = key.clone()

        if value is None:
            value = key.clone()

        # Transform and reshape the keys, values and queries
        # key = self.key_transform(key)
        # value = self.value_transform(value)
        # query = self.query_transform(query)

        if (
            self.narrow
        ):  # Implement narrow attention with aggregated context vectors per head

            # Transform and reshape the keys, values and queries
            key = self.get_dims_per_head(key)
            value = self.get_dims_per_head(value)
            query = self.get_dims_per_head(query)

            # Get dimensions of new shape
            batch_size, num_heads, seq_len, new_hidden_dim = query.shape

            # Generate context vectors from attention heads
            context_vectors = torch.stack(
                [
                    head(key=key, value=value, query=query, mask=mask)
                    for head in self.attention_heads
                ],
                dim=-1,
            )

            # Project context vectors back to original dimensions
            new_shape = batch_size, seq_len, num_heads * new_hidden_dim
            context_vectors = (
                context_vectors.sum(dim=-1).transpose(1, 2).contiguous().view(new_shape)
            )

        else:  # Implement wide attention with concatenated context vectors per head

            context_vectors = [
                head(key, value, query, mask=mask) for head in self.attention_heads
            ]

            context_vectors = torch.concat(context_vectors, dim=-1)

        # Persist the attention scores
        self.attention_scores = (
            torch.stack([head.attention_scores for head in self.attention_heads], dim=1)
            if not self.narrow
            else self.attention_heads[-1].attention_scores
        )

        print("From source: ", self.attention_scores.shape)
        print("Num heads: ", len(self.attention_heads))

        for i, head in enumerate(self.attention_heads):
            print(f"Attention head: {i}", head.attention_scores.shape)

        return self.context_transform(context_vectors)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        use_bias=False,
        dropout_probability=0.2,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self._attention_scores = None
        self.dropout_probability = dropout_probability

        self.narrow = narrow
        self.use_bias = use_bias

        self.key_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim**2, bias=use_bias
        )
        self.query_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim, bias=use_bias
        )

        self.attention_heads = [
            Attention(dropout_probability=dropout_probability)
            for _ in range(1 if self.narrow else num_heads)
        ]

        self.attention_heads = nn.ModuleList(self.attention_heads)

        if not narrow:
            self.context_transform = nn.Linear(
                in_features=sum([hidden_dim for _ in range(num_heads)]),
                out_features=self.hidden_dim,
                bias=True,
            )

        else:
            # self.query_transform = nn.LazyLinear(out_features=int(hidden_dim * 0.5))

            self.context_transform = nn.Linear(
                in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True
            )

    @property
    def attention_scores(self):
        return self._attention_scores

    @attention_scores.setter
    def attention_scores(self, new_attention_scores):
        self._attention_scores = new_attention_scores
        return

    def get_dims_per_head(self, x):
        N, L, H = x.shape
        x = x.view(N, L, self.num_heads, int(H / self.num_heads))
        return x.transpose(1, 2)

    def forward(self, key, query, mask=None):
        # Transform and reshape the keys, values and queries
        key, value = torch.split(
            self.key_transform(key), split_size_or_sections=key.size(-1) // 2, dim=-1
        )
        query = self.query_transform(query)

        if (
            self.narrow
        ):  # Implement narrow attention with aggregated context vectors per head

            # Transform and reshape the keys, values and queries
            key = self.get_dims_per_head(key)
            value = self.get_dims_per_head(value)
            query = self.get_dims_per_head(query)

            # Get dimensions of new shape
            batch_size, num_heads, seq_len, new_hidden_dim = query.shape

            # Generate context vectors from attention heads
            context_vectors = torch.stack(
                [
                    head(key=key, value=value, query=query, mask=mask)
                    for head in self.attention_heads
                ],
                dim=-1,
            )

            # Project context vectors back to original dimensions
            new_shape = batch_size, seq_len, num_heads * new_hidden_dim
            context_vectors = (
                context_vectors.sum(dim=-1).transpose(1, 2).contiguous().view(new_shape)
            )

        else:  # Implement wide attention with concatenated context vectors per head

            context_vectors = [
                head(key, value, query, mask=mask) for head in self.attention_heads
            ]

            context_vectors = torch.concat(context_vectors, dim=-1)

        # Persist the attention scores
        self.attention_scores = (
            torch.stack([head.attention_scores for head in self.attention_heads], dim=1)
            if not self.narrow
            else self.attention_heads[-1].attention_scores
        )

        return self.context_transform(context_vectors)

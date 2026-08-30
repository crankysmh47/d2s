from copy import deepcopy

import torch
from torch import nn

from ppmoe.moefication.reference import DenseSwiGLU, PartitionedSwiGLU


class TinySwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, heads: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.post_attention_norm = nn.RMSNorm(hidden_size)
        self.mlp: nn.Module = DenseSwiGLU(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden_states)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class TinySwiGLUTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                TinySwiGLUBlock(hidden_size, intermediate_size, heads)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_norm(hidden_states)


def partitioned_copy(
    model: TinySwiGLUTransformer,
    *,
    num_experts: int,
) -> TinySwiGLUTransformer:
    converted = deepcopy(model)
    for layer_index, layer in enumerate(converted.layers):
        dense = layer.mlp
        if not isinstance(dense, DenseSwiGLU):
            raise TypeError("expected a dense SwiGLU fixture")
        permutation = torch.roll(
            torch.arange(dense.intermediate_size),
            shifts=layer_index + 1,
        )
        partitions = tuple(permutation.reshape(num_experts, -1))
        layer.mlp = PartitionedSwiGLU.from_dense(dense, partitions)
    return converted


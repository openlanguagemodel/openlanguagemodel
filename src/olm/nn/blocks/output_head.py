from olm.nn.structure.block import Block
from olm.nn.norms import LayerNorm
from torch import nn
from torch.nn import functional as F
from olm.nn.torch_nn_wrappers import Linear
import torch


class _TiedEmbeddingProjection(nn.Module):
    """
    Linear language-model projection tied to an embedding matrix.

    The projection uses ``F.linear(x, embedding_weight)`` so the output logits
    share the same parameter as the token embedding table.
    """

    def __init__(self, embedding_weight: nn.Parameter, bias: bool = False) -> None:
        super().__init__()
        if embedding_weight.ndim != 2:
            raise ValueError(
                "Tied embedding weight must have shape (vocab_size, embed_dim)."
            )

        self.weight = embedding_weight
        self.out_features, self.in_features = embedding_weight.shape
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.out_features,
                    device=embedding_weight.device,
                    dtype=embedding_weight.dtype,
                )
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _resolve_embedding_weight(tied_embedding: nn.Module | nn.Parameter) -> nn.Parameter:
    if isinstance(tied_embedding, nn.Parameter):
        return tied_embedding

    embedding = getattr(tied_embedding, "embedding", tied_embedding)
    weight = getattr(embedding, "weight", None)
    if isinstance(weight, nn.Parameter):
        return weight

    raise TypeError(
        "tied_embedding must be an nn.Embedding, an OLM Embedding wrapper, "
        "or an embedding weight Parameter."
    )


class OutputHead(Block):
    """
    Final output projection layer for the Language Model.

    Consists of a LayerNorm followed by a projection to the vocabulary size.
    By default the projection has its own weights. Pass ``tied_embedding`` to
    share the projection matrix with the input token embedding.

    Args:
        embed_dim (int): The dimension of the embedding space.
        vocab_size (int): The size of the vocabulary.
        bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.
        tied_embedding (nn.Module | nn.Parameter, optional): Embedding module or
            weight parameter to reuse for the output projection.

    Attributes:
        layers (nn.ModuleList): The normalization and linear layers.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        bias: bool = False,
        tied_embedding=None,
    ):
        if tied_embedding is None:
            projection = Linear(embed_dim, vocab_size, bias=bias)
        else:
            embedding_weight = _resolve_embedding_weight(tied_embedding)
            projection = _TiedEmbeddingProjection(embedding_weight, bias=bias)
            if projection.in_features != embed_dim:
                raise ValueError(
                    "Tied embedding dimension does not match OutputHead embed_dim: "
                    f"{projection.in_features} != {embed_dim}."
                )
            if projection.out_features != vocab_size:
                raise ValueError(
                    "Tied embedding vocabulary size does not match OutputHead vocab_size: "
                    f"{projection.out_features} != {vocab_size}."
                )

        super().__init__([LayerNorm(embed_dim), projection])

    @property
    def projection(self) -> nn.Module:
        return self.blocks[1]

    @property
    def weight(self) -> nn.Parameter:
        return self.projection.weight

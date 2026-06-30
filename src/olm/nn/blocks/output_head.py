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
    Final normalization and vocabulary projection for language models.

    ``OutputHead`` applies a normalization layer and then maps hidden states to
    vocabulary logits. The projection is tied to the input token embedding by
    default. In the tied path, logits are computed as
    ``F.linear(hidden, embedding.weight)`` so the output head and token
    embedding share one parameter matrix.

    Forward:
        Accepts hidden states with shape ``[batch, seq_len, embed_dim]`` and
        returns logits with shape ``[batch, seq_len, vocab_size]``.

    Args:
        embed_dim (int): The dimension of the embedding space.
        vocab_size (int): The size of the vocabulary.
        bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.
        tied_embedding (nn.Module | nn.Parameter, optional): Embedding module or
            weight parameter to reuse for the output projection.
        tie_weights (bool, optional): Whether to reuse ``tied_embedding`` as
            the output projection matrix. Defaults to True.
        norm (nn.Module, optional): Normalization module before projection.
            Defaults to ``LayerNorm(embed_dim)``.
        use_norm (bool, optional): If False and ``norm`` is not provided, use
            an identity layer instead of LayerNorm. Defaults to True.

    Attributes:
        blocks (nn.ModuleList): ``[norm, projection]``.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        bias: bool = False,
        tied_embedding=None,
        tie_weights: bool = True,
        norm: nn.Module | None = None,
        use_norm: bool = True,
    ):
        if tie_weights and tied_embedding is None:
            raise ValueError(
                "OutputHead ties weights by default; pass tied_embedding=... "
                "or set tie_weights=False for an untied output matrix."
            )

        if not tie_weights:
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

        if norm is None:
            norm = LayerNorm(embed_dim) if use_norm else nn.Identity()

        super().__init__([norm, projection])

    @property
    def projection(self) -> nn.Module:
        """Projection module used after normalization."""
        return self.blocks[1]

    @property
    def weight(self) -> nn.Parameter:
        """Output projection weight; tied to the embedding matrix by default."""
        return self.projection.weight

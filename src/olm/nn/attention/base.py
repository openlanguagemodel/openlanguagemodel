import torch
import torch.nn as nn
from olm.nn.torch_nn_wrappers import Linear
from abc import ABC, abstractmethod
from typing import Optional
from olm.nn.embeddings.positional import RotaryPositionalEmbedding

class AttentionBase(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.

    Provides the common structure for attention layers, including QKV projections
    and output projection. Subclasses must implement the specific attention logic
    in `compute_attention`.

    Attributes:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        q_proj (Linear): Linear projection for Query.
        k_proj (Linear): Linear projection for Key.
        v_proj (Linear): Linear projection for Value.
        out_proj (Linear): Linear projection for Output.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        """
        Initializes the AttentionBase.

        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in linear projections. Defaults to True.

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # Shared QKV projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used

        # Shared output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used

    @abstractmethod
    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the attention scores and output.

        Args:
            q (torch.Tensor): Query tensor [batch, heads, seq, head_dim].
            k (torch.Tensor): Key tensor [batch, heads, seq, head_dim].
            v (torch.Tensor): Value tensor [batch, heads, seq, head_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: The attention output [batch, heads, seq, head_dim].
        """
        pass

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard forward pass for attention layers.

        Projects input to Q, K, V, calls `compute_attention`, and projects output.

        Args:
            x (torch.Tensor): Input tensor [batch, seq, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor [batch, seq, embed_dim].
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = self.compute_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

class AttentionwithRoPEBase(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms with Rotary Positional Embedding.

    Provides the common structure for attention layers, including QKV projections
    and output projection. Subclasses must implement the specific attention logic
    in `compute_attention`.

    Attributes:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        q_proj (Linear): Linear projection for Query.
        k_proj (Linear): Linear projection for Key.
        v_proj (Linear): Linear projection for Value.
        out_proj (Linear): Linear projection for Output.
    """
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, bias: bool = True):
        """
        Initializes the AttentionwithRoPEBase.

        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in linear projections. Defaults to True.

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.max_seq_len = max_seq_len

        # Shared QKV projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used

        # Shared output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used

    @abstractmethod
    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the attention scores and output.

        Args:
            q (torch.Tensor): Query tensor [batch, heads, seq, head_dim].
            k (torch.Tensor): Key tensor [batch, heads, seq, head_dim].
            v (torch.Tensor): Value tensor [batch, heads, seq, head_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: The attention output [batch, heads, seq, head_dim].
        """
        pass

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard forward pass for attention layers.

        Projects input to Q, K, V, calls `compute_attention`, and projects output.

        Args:
            x (torch.Tensor): Input tensor [batch, seq, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor [batch, seq, embed_dim].
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        k = self.rope(k)
        q = self.rope(q)

        out = self.compute_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.dropout(self.out_proj(out))

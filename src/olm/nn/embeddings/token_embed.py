import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS

class Embedding(nn.Module):
    """
    Token Embedding layer.

    Wraps standard PyTorch embedding with a clean interface.
    Maps integer indices to dense vectors.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the word embeddings.

    Attributes:
        embedding (nn.Embedding): The underlying PyTorch embedding layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the Embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        word_emb = self.embedding(x)
        return word_emb

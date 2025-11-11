# src/olm/nn/embeddings/token_embed.py
import torch, torch.nn as nn
from olm.core.registry import ACTIVATIONS

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the Embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        """
        Forward pass of the Embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        word_emb = self.embedding(x)
        return word_emb

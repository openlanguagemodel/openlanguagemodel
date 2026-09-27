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
        init_std (float): Standard deviation of the normal initializer. Defaults
            to 0.02, the value used by GPT-2, Llama, Qwen and Phi reference
            configs, and already used elsewhere in ``olm.nn``.

    Attributes:
        embedding (nn.Embedding): The underlying PyTorch embedding layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, init_std: float = 0.02):
        """
        Initialize the Embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
            init_std (float): Standard deviation of the normal initializer.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Applied once here, so a later ``self.embedding.reset_parameters()`` (the
        # usual meta-device/FSDP init path) would redraw N(0, 1) and undo this.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
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

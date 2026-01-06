from abc import ABC, abstractmethod
import torch
class TokenizerBase(ABC):
    """
    Abstract base class for all tokenizers in OLM.

    Defines the interface for converting between text strings and integer token IDs.
    Subclasses must implement `encode` and `decode` methods.
    """
    def __init__(self):
        """Initializes the tokenizer base."""
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Converts a text string into a sequence of token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            torch.Tensor: A 1D tensor containing the token IDs.
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Converts a sequence of token IDs back into a text string.

        Args:
            tokens (torch.Tensor): A 1D tensor or list of token IDs.

        Returns:
            str: The decoded text string.
        """
        pass

    def save(self, path: str) -> None:
        """
        Saves the tokenizer to a file.

        Args:
            path (str): Path to save the tokenizer to.

        Returns:
            None
        """
        pass

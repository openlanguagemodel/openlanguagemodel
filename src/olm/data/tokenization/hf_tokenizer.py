from typing import List
from olm.data.tokenization.base import TokenizerBase
from transformers import AutoTokenizer
import torch

class HFTokenizer(TokenizerBase):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single string into a 1D PyTorch tensor of input IDs. 
        Padding is implicitly disabled for single inputs.
        """
        encoded_data = self.tokenizer(
            text, 
            add_special_tokens=True, 
            return_tensors='pt', 
            # Padding is not needed for single strings, so we rely on default (False)
            truncation=False
        )
        # Squeeze to flatten the tensor from (1, N) to (N,)
        return encoded_data['input_ids'].squeeze(0)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decodes a single 1D tensor of token IDs back into a string."""
        
        # Squeeze and convert tensor to a 1D list of integers for decoding
        token_list: List[int] = tokens.squeeze().cpu().tolist()

        return self.tokenizer.decode(
            token_list, 
            skip_special_tokens=True
        )
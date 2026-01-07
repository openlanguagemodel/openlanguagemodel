from typing import List
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from olm.data.tokenization.base import TokenizerBase

class HFTokenizerTrainCustom(TokenizerBase):
    def __init__(self, files: List[str], vocab_size: int, special_tokens: List[str], save_location: str, unk_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(save_location)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single string into a 1D PyTorch tensor of input IDs. 
        Padding is implicitly disabled for single inputs.
        """
        encoding = self.tokenizer.encode(text)
        # encoding.ids is a list of token IDs
        return torch.tensor(encoding.ids, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decodes a single 1D tensor of token IDs back into a string."""
        
        # Squeeze and convert tensor to a 1D list of integers for decoding
        token_list: List[int] = tokens.squeeze().cpu().tolist()

        return self.tokenizer.decode(token_list)
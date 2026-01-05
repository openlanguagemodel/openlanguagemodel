import torch
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union, Optional, Any
from torch.utils.data import IterableDataset, get_worker_info
from itertools import islice

class BaseTextDataset(IterableDataset, ABC):
    """
    Abstract base class for text-based streaming datasets.
    
    Handles tokenization buffering and sequence generation generically.
    Subclasses must implement `_get_text_iterator` to yield text chunks.
    """
    def __init__(
        self, 
        tokenizer: Any, 
        context_length: int, 
        skip_batches: int = 0,
        shuffle: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.skip_batches = skip_batches
        self.shuffle = shuffle
        self.seed = seed
        self.token_buffer = []

    @abstractmethod
    def _get_text_iterator(self) -> Iterator[str]:
        """
        Yields strings of text to be tokenized and buffered.
        The size of these strings does not matter; they will be concatenated.
        """
        pass

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over tokens, yielding (input, target) tensors.
        Handles multi-worker sharding and buffering.
        """
        worker_info = get_worker_info()
        batches_yielded = 0
        
        # Get the text source
        text_iter = self._get_text_iterator()

        # Handle multi-worker sharding
        if worker_info is not None:
            # Each worker only processes a subset of the text chunks
            # to avoid duplication across workers.
            text_iter = islice(text_iter, worker_info.id, None, worker_info.num_workers)

        for text in text_iter:
            # Generic encoding
            try:
                # Try to use add_special_tokens=False for continuous streaming
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                # Fallback for simple tokenizers
                tokens = self.tokenizer.encode(text)

            self.token_buffer.extend(tokens)

            # Yield complete sequences
            while len(self.token_buffer) >= self.context_length + 1:
                # Extract sequence
                sequence = self.token_buffer[: self.context_length + 1]
                
                # Non-overlapping sliding window
                self.token_buffer = self.token_buffer[self.context_length + 1 :]

                # Skip batches if needed (e.g. resumption)
                if batches_yielded < self.skip_batches:
                    batches_yielded += 1
                    continue

                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                labels = torch.tensor(sequence[1:], dtype=torch.long)
                
                batches_yielded += 1
                yield input_ids, labels

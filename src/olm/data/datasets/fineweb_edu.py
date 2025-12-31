# src/olm/data/datasets/fineweb_edu.py
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Optional, Iterator, Tuple
import numpy as np
from transformers import GPT2TokenizerFast
from src.olm.data.datasets import DataLoader


class FineWebEduDataset(IterableDataset):
    """
    FineWeb Edu dataset for GPT-2 training.

    This dataset streams data from HuggingFace's FineWeb-Edu dataset,
    which contains high-quality educational web pages.

    Args:
        split: Dataset split ('train' or 'validation')
        context_length: Sequence length for training (default: 1024)
        subset: Dataset subset to use (default: 'sample-10BT' for 10B tokens)
        streaming: Whether to use streaming mode (default: True)
        cache_dir: Directory to cache downloaded data (default: None)
        tokenizer_name: Name of tokenizer to use (default: 'gpt2')

    Example:
        >>> dataset = FineWebEduDataset(split='train', context_length=1024)
        >>> for input_ids, labels in dataset:
        >>>     # input_ids and labels are both [context_length] tensors
        >>>     pass
    """

    def __init__(
        self,
        split: str = "train",
        context_length: int = 1024,
        subset: str = "sample-10BT",
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        tokenizer_name: str = "gpt2",
    ):
        super().__init__()
        self.split = split
        self.context_length = context_length
        self.subset = subset
        self.streaming = streaming
        self.cache_dir = cache_dir

        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

        # Load dataset
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=subset,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )

        # Buffer for partial sequences
        self.token_buffer = []

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over tokenized sequences.

        Yields:
            Tuple of (input_ids, labels) where both are [context_length] tensors.
            Labels are shifted by 1 position (next token prediction).
        """
        for example in self.dataset:
            # Tokenize the text
            text = example["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Add to buffer
            self.token_buffer.extend(tokens)

            # Yield complete sequences
            while len(self.token_buffer) >= self.context_length + 1:
                # Extract sequence of length context_length + 1
                sequence = self.token_buffer[: self.context_length + 1]
                self.token_buffer = self.token_buffer[self.context_length :]

                # Create input and target
                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                labels = torch.tensor(sequence[1:], dtype=torch.long)

                yield input_ids, labels

    def __len__(self):
        """
        Length is not well-defined for streaming datasets.
        Returns an estimate based on the subset size.
        """
        # FineWeb-Edu sample-10BT has ~10B tokens
        # With context_length sequences, that's approximately:
        if self.subset == "sample-10BT":
            total_tokens = 10_000_000_000
        else:
            # Conservative estimate
            total_tokens = 1_000_000_000

        return total_tokens // self.context_length


class FineWebEduDataLoader:
    """
    DataLoader wrapper for FineWebEduDataset with multi-GPU support.

    This handles batching and distributed sampling for the streaming dataset.

    Args:
        dataset: FineWebEduDataset instance
        batch_size: Batch size per device
        num_workers: Number of data loading workers (default: 0)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
    """

    def __init__(
        self,
        dataset: FineWebEduDataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Create PyTorch DataLoader

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)

    def __len__(self):
        """Return number of batches."""
        return len(self.dataset) // self.batch_size

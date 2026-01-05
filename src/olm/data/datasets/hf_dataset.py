# src/olm/data/datasets/hf_dataset.py
import torch
from datasets import load_dataset
from typing import Optional, Iterator, Callable, Any, Dict
from olm.data.datasets.base_dataset import BaseTextDataset


class HuggingFaceTextDataset(BaseTextDataset):
    """
    Generic dataset loader for Hugging Face text datasets.
    
    Inherits from BaseTextDataset to share token buffering logic and multi-worker safety.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        context_length: int,
        text_fn: Callable[[Any], str],
        tokenizer: Any,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        streaming: bool = True,
        skip_batches: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer_size: int = 10_000,
    ):
        super().__init__(
            tokenizer=tokenizer, 
            context_length=context_length, 
            skip_batches=skip_batches,
            shuffle=shuffle,
            seed=seed
        )

        self.dataset_name = dataset_name
        self.split = split
        self.text_fn = text_fn
        self.dataset_kwargs = dataset_kwargs or {}
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size

        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            **self.dataset_kwargs,
        )

        # Apply shuffling if requested
        if self.shuffle:
            # For streaming datasets, shuffle requires a buffer_size
            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

    def _get_text_iterator(self) -> Iterator[str]:
        """Yields text from the Hugging Face dataset."""
        for example in self.dataset:
            text = self.text_fn(example)
            if text:
                yield text


class FineWebEduDataset(HuggingFaceTextDataset):
    """
    FineWeb Edu dataset configuration.

    Args:
        split: Dataset split ('train' or 'validation')
        context_length: Sequence length for training (default: 1024)
        subset: Dataset subset to use (default: 'sample-10BT')
        tokenizer: Tokenizer object (e.g. from AutoTokenizer)
        streaming: Whether to use streaming mode (default: True)
        shuffle: Whether to shuffle the dataset (default: False)
        seed: Random seed for shuffling (default: 42)
        cache_dir: Directory to cache downloaded data (default: None)
        skip_batches: Number of batches to skip
    """

    def __init__(
        self,
        tokenizer: Any,
        split: str = "train",
        context_length: int = 1024,
        subset: str = "sample-10BT",
        streaming: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        skip_batches: int = 0,
    ):
        self.subset = subset
        super().__init__(
            dataset_name="HuggingFaceFW/fineweb-edu",
            split=split,
            context_length=context_length,
            text_fn=lambda ex: ex["text"],
            tokenizer=tokenizer,
            dataset_kwargs={"name": subset, "cache_dir": cache_dir},
            streaming=streaming,
            skip_batches=skip_batches,
            shuffle=shuffle,
            seed=seed
        )

    def __len__(self):
        """Estimate based on subset size."""
        if self.subset == "sample-10BT":
            total_tokens = 10_000_000_000
        else:
            total_tokens = 1_000_000_000

        return total_tokens // self.context_length

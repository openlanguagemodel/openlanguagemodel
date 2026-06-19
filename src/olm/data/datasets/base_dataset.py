import torch
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any
from torch.utils.data import IterableDataset, get_worker_info
from itertools import islice


class BaseTextDataset(IterableDataset, ABC):
    """
    Abstract base class for text-based streaming datasets.

    ``BaseTextDataset`` handles tokenization, buffering, next-token target
    construction, worker sharding, and distributed-rank sharding. Subclasses
    only need to implement ``_get_text_iterator`` and yield raw text strings.

    Iteration:
        Yields ``(input_ids, labels)`` tuples. Both tensors have shape
        ``[context_length]`` and dtype ``torch.long``. ``labels`` is the
        one-token-shifted target sequence for causal language modeling.

    Args:
        tokenizer: Tokenizer with an ``encode`` method.
        context_length (int): Number of input tokens per sample.
        skip_batches (int): Number of yielded samples to skip, useful for
            coarse resume behavior.
        shuffle (bool): Whether the concrete dataset should shuffle its source
            stream when supported.
        seed (int): Shuffle seed.
    """

    def __init__(
        self,
        tokenizer: Any,
        context_length: int,
        skip_batches: int = 0,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.skip_batches = skip_batches
        self.shuffle = shuffle
        self.seed = seed

    @abstractmethod
    def _get_text_iterator(self) -> Iterator[str]:
        """
        Yield raw text chunks to be tokenized and buffered.

        Returns:
            Iterator[str]: Text fragments. Fragment size does not matter because
            ``BaseTextDataset`` concatenates tokens into a rolling buffer.
        """
        pass

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over tokens, yielding (input, target) tensors.

        Handles distributed-rank and multi-worker sharding and buffering.

        Returns:
            Iterator[tuple[torch.Tensor, torch.Tensor]]: ``input_ids`` and
            ``labels`` tensors, each shaped ``[context_length]``.
        """
        worker_info = get_worker_info()
        batches_yielded = 0
        token_buffer = []

        # Get the text source
        text_iter = self._get_text_iterator()

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # Handle distributed-rank and multi-worker sharding together so iterable
        # datasets do not replay the same text on every rank.
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        if num_shards > 1:
            text_iter = islice(text_iter, shard_id, None, num_shards)

        for text in text_iter:
            # Generic encoding
            try:
                # Try to use add_special_tokens=False for continuous streaming
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                # Fallback for simple tokenizers
                tokens = self.tokenizer.encode(text)

            token_buffer.extend(tokens)

            # Yield complete sequences
            while len(token_buffer) >= self.context_length + 1:
                # Extract sequence
                sequence = token_buffer[: self.context_length + 1]

                # Non-overlapping sliding window
                token_buffer = token_buffer[self.context_length + 1 :]

                # Skip batches if needed (e.g. resumption)
                if batches_yielded < self.skip_batches:
                    batches_yielded += 1
                    continue

                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                labels = torch.tensor(sequence[1:], dtype=torch.long)

                batches_yielded += 1
                yield input_ids, labels

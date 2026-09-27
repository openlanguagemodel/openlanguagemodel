import torch
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Any
from torch.utils.data import IterableDataset, get_worker_info
from itertools import islice


class BaseTextDataset(IterableDataset, ABC):
    """
    Abstract base class for text-based streaming datasets.

    ``BaseTextDataset`` handles tokenization, buffering, next-token target
    construction, worker sharding, and distributed-rank sharding. Subclasses
    only need to implement ``_get_text_iterator`` and yield raw text strings.

    Tokens from consecutive texts go into one buffer, so a sample can span the
    end of one text and the start of the next. Each sample takes
    ``context_length + 1`` fresh tokens from the buffer, and tokens left over
    when the source runs out are dropped.

    Iteration:
        Yields ``(input_ids, labels)`` tuples. Both tensors have shape
        ``[context_length]`` and dtype ``torch.long``. ``labels`` is the
        one-token-shifted target sequence for causal language modeling.

    Args:
        tokenizer: Tokenizer with an ``encode`` method that returns a 1-D
            tensor or a list of token ids.
        context_length (int): Number of input tokens per sample.
        skip_batches (int): Number of yielded samples to skip, useful for
            coarse resume behavior. The count applies to each worker and rank
            shard separately.
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

    def _source_splits_across_workers(self) -> bool:
        """
        Whether the text source already gives each DataLoader worker its own data.

        When this returns ``True``, ``__iter__`` shards only across distributed
        ranks and leaves the worker split to the source. Override it for sources
        that split themselves, or every worker would keep only part of its share.
        """
        return False

    def _encode(self, text: str) -> List[int]:
        """Tokenize one text chunk, asking for no special tokens, as a list of ids."""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            # Older/simple tokenizers may not expose add_special_tokens.
            tokens = self.tokenizer.encode(text)

        # Extending a list with a tensor would store one 0-d tensor per token.
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return tokens

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
        sample_length = self.context_length + 1

        # Get the text source
        text_iter = self._get_text_iterator()

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # Handle distributed-rank and multi-worker sharding together so iterable
        # datasets do not replay the same text on every rank.
        if worker_info is not None and not self._source_splits_across_workers():
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
            token_buffer.extend(self._encode(text))

            # Yield complete sequences. ``start`` walks through the buffer, and
            # the used tokens are removed once per text, so a long text costs
            # linear rather than quadratic time.
            start = 0
            while len(token_buffer) - start >= sample_length:
                # Non-overlapping sliding window
                sequence = token_buffer[start : start + sample_length]
                start += sample_length

                # Skip batches if needed (e.g. resumption)
                if batches_yielded < self.skip_batches:
                    batches_yielded += 1
                    continue

                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                labels = torch.tensor(sequence[1:], dtype=torch.long)

                batches_yielded += 1
                yield input_ids, labels

            del token_buffer[:start]

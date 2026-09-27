# src/olm/data/datasets/hf_dataset.py
import torch
from typing import Optional, Iterator, Callable, Any, Dict
from olm.data.datasets.base_dataset import BaseTextDataset


def _import_load_dataset():
    """
    Import ``datasets.load_dataset`` on first use.

    Only the Hugging Face dataset classes need the ``datasets`` package, so a
    missing or broken install breaks them alone and not ``import olm``.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "HuggingFaceTextDataset and FineWebEduDataset need the Hugging Face "
            "`datasets` package, which could not be imported "
            f"({type(exc).__name__}: {exc}). Install or repair it with "
            "`pip install -U datasets`."
        ) from exc
    return load_dataset


class HuggingFaceTextDataset(BaseTextDataset):
    """
    Generic dataset loader for Hugging Face text datasets.

    The Hugging Face dataset is read in streaming mode by default, then text is
    extracted with ``text_fn`` and passed through ``BaseTextDataset`` for
    tokenization, sharding, and next-token target construction.

    Needs the Hugging Face ``datasets`` package, which is imported when the
    dataset is created, not when ``olm`` is imported.

    Implementation note:
        A streaming ``datasets.IterableDataset`` gives each DataLoader worker
        its own subset of the source files, so this class leaves the worker
        split to ``datasets`` and only splits across distributed ranks itself.
        Workers beyond the number of source shards (``self.dataset.n_shards``)
        get no files and yield nothing. A shuffled stream reports fewer shards
        than it has files, because ``datasets`` mixes groups of files into its
        shuffle buffer.

    Iteration:
        Yields ``(input_ids, labels)`` tensors shaped ``[context_length]``.

    Args:
        dataset_name (str): Hugging Face dataset name.
        split (str): Dataset split, such as ``"train"``.
        context_length (int): Number of input tokens per sample.
        text_fn: Function that maps a dataset example to text.
        tokenizer: Tokenizer with an ``encode`` method.
        dataset_kwargs: Extra keyword arguments passed to ``load_dataset``.
        streaming (bool): Whether to use Hugging Face streaming mode.
        skip_batches (int): Number of samples to skip before yielding.
        shuffle (bool): Whether to shuffle the dataset stream.
        seed (int): Shuffle seed.
        shuffle_buffer_size (int): Streaming shuffle buffer size. Not used when
            ``streaming=False``, where the whole split is shuffled.
        append_eos (bool): Whether to add the tokenizer's EOS token after each
            example, marking where one example ends and the next begins.

    Raises:
        ImportError: If the ``datasets`` package cannot be imported.
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
        append_eos: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            context_length=context_length,
            skip_batches=skip_batches,
            shuffle=shuffle,
            seed=seed,
            append_eos=append_eos,
        )

        self.dataset_name = dataset_name
        self.split = split
        self.text_fn = text_fn
        self.dataset_kwargs = dataset_kwargs or {}
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size

        # Load dataset
        load_dataset = _import_load_dataset()
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            **self.dataset_kwargs,
        )

        # Apply shuffling if requested
        if self.shuffle and self.streaming:
            # For streaming datasets, shuffle requires a buffer_size
            self.dataset = self.dataset.shuffle(
                seed=self.seed, buffer_size=self.shuffle_buffer_size
            )
        elif self.shuffle:
            # An in-memory ``datasets.Dataset`` shuffles its whole index instead
            self.dataset = self.dataset.shuffle(seed=self.seed)

    def _source_splits_across_workers(self) -> bool:
        """Whether ``datasets`` splits this dataset across DataLoader workers."""
        # ``datasets`` makes its streaming ``IterableDataset`` a subclass of the
        # PyTorch one and, inside a DataLoader worker, iterates only that
        # worker's files. This is the same check it makes before doing so.
        return isinstance(self.dataset, torch.utils.data.IterableDataset)

    def _get_text_iterator(self) -> Iterator[str]:
        """Yields text from the Hugging Face dataset."""
        for example in self.dataset:
            text = self.text_fn(example)
            if text:
                yield text


class FineWebEduDataset(HuggingFaceTextDataset):
    """
    Convenience wrapper for ``HuggingFaceFW/fineweb-edu``.

    Iteration:
        Yields ``(input_ids, labels)`` tensors shaped ``[context_length]`` for
        causal language-model training.

    Args:
        tokenizer: Tokenizer with an ``encode`` method.
        split: Dataset split, such as ``"train"``.
        context_length: Sequence length for training (default: 1024)
        subset: Dataset subset to use (default: 'sample-10BT')
        streaming: Whether to use streaming mode (default: True)
        shuffle: Whether to shuffle the dataset (default: False)
        seed: Random seed for shuffling (default: 42)
        cache_dir: Directory to cache downloaded data (default: None)
        skip_batches: Number of batches to skip
        append_eos: Whether to add the tokenizer's EOS token after each
            document (default: False). New runs should usually set it, so the
            model learns where documents end.
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
        append_eos: bool = False,
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
            seed=seed,
            append_eos=append_eos,
        )

    def __len__(self):
        """Estimate based on subset size."""
        if self.subset == "sample-10BT":
            total_tokens = 10_000_000_000
        else:
            total_tokens = 1_000_000_000

        return total_tokens // self.context_length

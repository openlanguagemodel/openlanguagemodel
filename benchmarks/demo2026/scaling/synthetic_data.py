"""Fixed synthetic token batches for throughput benchmarks."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


class SyntheticTokenRing(IterableDataset):
    """Infinite iterator over a pre-generated ring of (input_ids, labels) pairs.

    Labels are ``input_ids.roll(-1)`` so the training loss matches the OLM
    trainer contract without a tokenizer or disk I/O.
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        ring_size: int,
        seed: int,
    ):
        generator = torch.Generator().manual_seed(seed)
        ring = torch.randint(
            0,
            vocab_size,
            (ring_size, sequence_length),
            generator=generator,
        )
        self.inputs = ring
        self.labels = torch.roll(ring, shifts=-1, dims=1)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        idx = 0
        n = self.inputs.size(0)
        while True:
            yield self.inputs[idx % n], self.labels[idx % n]
            idx += 1

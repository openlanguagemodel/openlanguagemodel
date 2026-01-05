import os
import torch
import random
from typing import Iterator, Union
from olm.data.datasets.base_dataset import BaseTextDataset


class LocalTextDataset(BaseTextDataset):
    """
    Dataset that streams text from local .txt files in a directory.
    """
    def __init__(
        self,
        location: Union[str, os.PathLike],
        tokenizer,
        context_length: int,
        skip_batches: int = 0,
        shuffle: bool = False,
        seed: int = 42
    ):
        super().__init__(
            tokenizer=tokenizer, 
            context_length=context_length, 
            skip_batches=skip_batches,
            shuffle=shuffle,
            seed=seed
        )
        self.location = os.fspath(location)
        
        # Discover files - initial sort for determinism
        try:
            self.files = sorted(
                f for f in os.listdir(self.location)
                if f.endswith(".txt")
                and os.path.isfile(os.path.join(self.location, f))
            )
        except FileNotFoundError:
             self.files = []

    def _get_text_iterator(self) -> Iterator[str]:
        files_to_process = list(self.files)
        
        if self.shuffle:
            # Use fixed seed for reproducibility within an epoch
            rng = random.Random(self.seed)
            rng.shuffle(files_to_process)

        for fname in files_to_process:
            path = os.path.join(self.location, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): 
                        yield line

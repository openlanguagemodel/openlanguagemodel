import os
import torch
from torch.utils.data import IterableDataset
from typing import Union

class Dataset(IterableDataset):
    def __init__(
        self,
        location: Union[str, os.PathLike],
        tokenizer,
        context_length: int,
    ):
        self.location = os.fspath(location)
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.files = sorted(
            f for f in os.listdir(self.location)
            if f.endswith(".txt")
            and os.path.isfile(os.path.join(self.location, f))
        )

    def __iter__(self):
        token_buffer = []

        for fname in self.files:
            path = os.path.join(self.location, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = self.tokenizer.encode(line)
                    token_buffer.extend(tokens)

                    while len(token_buffer) >= self.context_length + 1:
                        chunk = token_buffer[: self.context_length + 1]
                        token_buffer = token_buffer[self.context_length :]

                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y

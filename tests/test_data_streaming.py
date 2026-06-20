import torch

from olm.data.datasets.base_dataset import BaseTextDataset


class NumberTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [int(token) for token in text.split()]


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens)
        return [int(token) for token in text.split()]


class ErrorTokenizer:
    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        raise ValueError("tokenizer failed")


class NumberDataset(BaseTextDataset):
    def __init__(self, chunks, tokenizer=None, **kwargs):
        super().__init__(
            tokenizer=tokenizer or NumberTokenizer(),
            context_length=2,
            **kwargs,
        )
        self.chunks = chunks

    def _get_text_iterator(self):
        yield from self.chunks


def test_iterable_dataset_shards_across_distributed_ranks(monkeypatch):
    chunks = ["0 1 2", "10 11 12", "20 21 22", "30 31 32"]
    dataset = NumberDataset(chunks)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    samples = list(dataset)

    assert [sample[0].tolist() for sample in samples] == [[10, 11], [30, 31]]
    assert [sample[1].tolist() for sample in samples] == [[11, 12], [31, 32]]


def test_iterable_dataset_buffer_is_per_iterator():
    dataset = NumberDataset(["0 1 2 3"])

    first = list(dataset)
    second = list(dataset)

    assert [sample[0].tolist() for sample in first] == [[0, 1]]
    assert [sample[0].tolist() for sample in second] == [[0, 1]]


def test_streaming_dataset_disables_special_tokens():
    tokenizer = RecordingTokenizer()
    dataset = NumberDataset(["0 1 2"], tokenizer=tokenizer)

    list(dataset)

    assert tokenizer.calls == [False]


def test_streaming_dataset_does_not_swallow_tokenizer_errors():
    dataset = NumberDataset(["0 1 2"], tokenizer=ErrorTokenizer())

    try:
        list(dataset)
    except ValueError as exc:
        assert str(exc) == "tokenizer failed"
    else:
        raise AssertionError("expected tokenizer error to propagate")

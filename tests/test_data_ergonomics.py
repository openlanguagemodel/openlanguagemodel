import importlib.abc
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

import olm
from olm.data.datasets import (
    FineWebEduDataset,
    HuggingFaceTextDataset,
    LocalTextDataset,
)
from olm.data.datasets import hf_dataset
from olm.data.datasets.base_dataset import BaseTextDataset
from olm.data.tokenization import HFTokenizer

# Base vocabulary of the in-test tokenizer: "<unk>" is id 0, "0".."19" are ids
# 1..20, and "<eos>" is added on top as a special token with id 21.
WORDS = ["<unk>"] + [str(i) for i in range(20)]
EOS_ID = len(WORDS)


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A tiny word-level tokenizer saved in Hugging Face format, built offline."""
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    fast.add_special_tokens({"eos_token": "<eos>"})

    path = tmp_path_factory.mktemp("tokenizer")
    fast.save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def tok(tokenizer_dir):
    return HFTokenizer(tokenizer_dir)


def run_python(*args):
    """Run a fresh interpreter that imports the same ``olm`` as this test."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(olm.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, os.environ.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, timeout=300, env=env
    )


def number_text(example):
    return example["text"]


class NumberTokenizer:
    """Whitespace tokenizer over integers, returning tensors like HFTokenizer."""

    eos_token_id = 99

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return torch.tensor([int(token) for token in text.split()])


class NumberDataset(BaseTextDataset):
    def __init__(self, chunks, tokenizer=None, context_length=2, **kwargs):
        super().__init__(
            tokenizer=tokenizer or NumberTokenizer(),
            context_length=context_length,
            **kwargs,
        )
        self.chunks = chunks

    def _get_text_iterator(self):
        yield from self.chunks


def write_text_files(root, num_files=4, lines_per_file=25):
    """Write ``num_files`` files of three-number lines. Returns their paths."""
    paths = []
    for f in range(num_files):
        path = os.path.join(root, f"part{f}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for line in range(lines_per_file):
                start = f * lines_per_file + line
                handle.write(" ".join(str((start + k) % 20) for k in range(3)))
                handle.write("\n")
        paths.append(path)
    return paths


def as_lists(samples):
    return [(x.tolist(), y.tolist()) for x, y in samples]


# --- HFTokenizer ---


def test_encode_returns_1d_long_tensor_by_default(tok):
    ids = tok.encode("1 2 3")

    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    assert ids.shape == (3,)
    assert ids.tolist() == [2, 3, 4]


def test_encode_return_tensors_none_returns_list(tok):
    ids = tok.encode("1 2 3", return_tensors=None)

    assert ids == [2, 3, 4]
    assert all(type(i) is int for i in ids)


def test_encode_rejects_unknown_return_tensors(tok):
    with pytest.raises(ValueError, match="return_tensors"):
        tok.encode("1 2", return_tensors="np")


def test_encode_works_when_transformers_disables_pytorch(tok, monkeypatch):
    # ``transformers`` switches off its PyTorch support when the installed torch
    # is older than it supports, and ``return_tensors="pt"`` then fails.
    from transformers import tokenization_utils_base

    monkeypatch.setattr(tokenization_utils_base, "is_torch_available", lambda: False)
    with pytest.raises(ImportError):
        tok.tokenizer("1 2", return_tensors="pt")

    ids = tok.encode("1 2")

    assert isinstance(ids, torch.Tensor)
    assert ids.tolist() == [2, 3]


def test_vocab_size_counts_added_special_tokens(tok):
    ids = tok.encode("1 <eos>")

    assert ids.tolist() == [2, EOS_ID]
    assert tok.vocab_size == len(tok.tokenizer) == EOS_ID + 1
    # An embedding sized with vocab_size must accept every id encode returns.
    torch.nn.Embedding(tok.vocab_size, 4)(ids)


def test_special_token_ids(tok):
    assert tok.eos_token_id == EOS_ID
    assert tok.bos_token_id is None


def test_decode_accepts_tensors_lists_and_single_ids(tok):
    ids = tok.encode("1 2 3")

    assert tok.decode(ids) == "1 2 3"
    assert tok.decode(ids.unsqueeze(0)) == "1 2 3"
    assert tok.decode(ids.tolist()) == "1 2 3"
    assert tok.decode(ids[:1]) == "1"
    assert tok.decode(2) == "1"
    # One sampled id, as ``[1, 1]`` from batched sampling or as a 0-d tensor.
    assert tok.decode(ids[:1].view(1, 1)) == "1"
    assert tok.decode(ids[0]) == "1"


def test_decode_skips_special_tokens_unless_asked(tok):
    ids = tok.encode("1 <eos>")

    assert tok.decode(ids) == "1"
    assert tok.decode(ids, skip_special_tokens=False) == "1 <eos>"


def test_decode_rejects_a_batch(tok):
    batch = torch.stack([tok.encode("1 2"), tok.encode("3 4")])

    with pytest.raises(ValueError, match="one sequence"):
        tok.decode(batch)
    # A ``[batch, 1]`` column holds one token per sequence, not one sequence.
    with pytest.raises(ValueError, match="one sequence"):
        tok.decode(batch[:, :1])


def test_save_and_load_round_trip(tok, tmp_path):
    tok.save(str(tmp_path))
    loaded = HFTokenizer.load(str(tmp_path))

    assert (tmp_path / "type").read_text() == "HFTokenizer"
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.eos_token_id == tok.eos_token_id
    assert loaded.encode("1 2 <eos>").tolist() == tok.encode("1 2 <eos>").tolist()


# --- Optional dependencies ---


class _BrokenImport(importlib.abc.MetaPathFinder):
    """Makes importing one package fail the way a broken install can."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path, target=None):
        if fullname == self.name:
            raise AttributeError(f"simulated broken {self.name} install")
        return None


def test_hf_dataset_reports_missing_datasets_package(tok, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(ImportError, match="pip install -U datasets"):
        HuggingFaceTextDataset(
            dataset_name="text",
            split="train",
            context_length=2,
            text_fn=number_text,
            tokenizer=tok,
        )
    with pytest.raises(ImportError, match="FineWebEduDataset"):
        FineWebEduDataset(tok)


def test_hf_dataset_reports_broken_datasets_package(tok, monkeypatch):
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenImport("datasets")] + sys.meta_path)

    with pytest.raises(ImportError, match="simulated broken datasets install"):
        FineWebEduDataset(tok)


def test_hf_tokenizer_reports_missing_transformers_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match="pip install -U transformers"):
        HFTokenizer("gpt2")


def test_import_olm_without_datasets_or_transformers(tmp_path):
    # A fresh interpreter, so the check does not depend on what this test
    # session has imported already.
    script = textwrap.dedent(
        """
        import json, sys
        sys.modules["datasets"] = None
        sys.modules["transformers"] = None

        import olm
        import olm.train
        from olm.data.datasets import FineWebEduDataset, LocalTextDataset
        from olm.data.tokenization import HFTokenizer


        class Chars:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]


        samples = list(LocalTextDataset(sys.argv[1], Chars(), context_length=3))
        try:
            FineWebEduDataset(Chars())
            error = None
        except ImportError as exc:
            error = str(exc)
        print(json.dumps({"samples": len(samples), "error": error}))
        """
    )
    (tmp_path / "a.txt").write_text("abcdefgh\n", encoding="utf-8")
    result = run_python("-c", script, str(tmp_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["samples"] == 2
    assert "pip install -U datasets" in report["error"]


def test_import_olm_does_not_import_datasets_or_transformers():
    script = (
        "import json, sys, olm, olm.train; "
        "print(json.dumps([m in sys.modules for m in ('datasets', 'transformers')]))"
    )
    result = run_python("-c", script)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [False, False]


# --- Packing ---


def test_dataset_accepts_tensor_and_list_tokens():
    class ListTokenizer(NumberTokenizer):
        def encode(self, text, add_special_tokens=True):
            return super().encode(text).tolist()

    chunks = ["0 1 2 3 4 5"]
    from_tensors = list(NumberDataset(chunks))
    from_lists = list(NumberDataset(chunks, tokenizer=ListTokenizer()))

    assert as_lists(from_tensors) == as_lists(from_lists)
    assert as_lists(from_tensors) == [([0, 1], [1, 2]), ([3, 4], [4, 5])]
    for x, y in from_tensors:
        assert x.dtype == y.dtype == torch.long


def test_long_text_is_split_into_consecutive_windows():
    tokens = [i % 50 for i in range(10_000)]
    samples = list(
        NumberDataset([" ".join(map(str, tokens))], context_length=7, skip_batches=3)
    )

    assert len(samples) == 10_000 // 8 - 3
    for n, (x, y) in enumerate(samples, start=3):
        window = tokens[n * 8 : (n + 1) * 8]
        assert x.tolist() == window[:-1]
        assert y.tolist() == window[1:]


def test_append_eos_separates_texts():
    dataset = NumberDataset(["0 1", "2 3"], append_eos=True)

    assert as_lists(dataset) == [([0, 1], [1, 99]), ([2, 3], [3, 99])]


def test_append_eos_is_off_by_default():
    dataset = NumberDataset(["0 1", "2 3 4 5"])

    assert as_lists(dataset) == [([0, 1], [1, 2]), ([3, 4], [4, 5])]


def test_append_eos_needs_a_tokenizer_with_eos():
    class NoEos:
        def encode(self, text, add_special_tokens=False):
            return [int(t) for t in text.split()]

    with pytest.raises(ValueError, match="eos_token_id"):
        NumberDataset(["0 1"], tokenizer=NoEos(), append_eos=True)


def test_local_dataset_appends_hf_tokenizer_eos(tok, tmp_path):
    (tmp_path / "a.txt").write_text("1 2\n3 4\n", encoding="utf-8")
    dataset = LocalTextDataset(tmp_path, tok, context_length=2, append_eos=True)

    assert as_lists(dataset) == [([2, 3], [3, EOS_ID]), ([4, 5], [5, EOS_ID])]


def test_fineweb_edu_passes_subset_and_append_eos(tok, monkeypatch):
    calls = []

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return [{"text": "1 2"}, {"text": ""}, {"text": "3 4"}]

    monkeypatch.setattr(hf_dataset, "_import_load_dataset", lambda: fake_load_dataset)
    dataset = FineWebEduDataset(
        tok, context_length=2, subset="sample-10BT", append_eos=True
    )

    assert calls == [
        (
            "HuggingFaceFW/fineweb-edu",
            {
                "split": "train",
                "streaming": True,
                "name": "sample-10BT",
                "cache_dir": None,
            },
        )
    ]
    assert as_lists(dataset) == [([2, 3], [3, EOS_ID]), ([4, 5], [5, EOS_ID])]


# --- Hugging Face datasets, offline from local text files ---


def make_hf_dataset(tok, files, streaming, cache_dir, **kwargs):
    return HuggingFaceTextDataset(
        dataset_name="text",
        split="train",
        context_length=2,
        text_fn=number_text,
        tokenizer=tok,
        dataset_kwargs={"data_files": files, "cache_dir": cache_dir},
        streaming=streaming,
        **kwargs,
    )


@pytest.mark.parametrize("streaming", [True, False])
def test_dataloader_workers_split_hf_dataset_without_dropping_data(
    tok, tmp_path, streaming
):
    files = write_text_files(str(tmp_path))
    dataset = make_hf_dataset(tok, files, streaming, str(tmp_path / "cache"))

    def collect(num_workers):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=num_workers
        )
        return sorted(as_lists(loader))

    # 4 files of 25 three-token lines: 300 tokens, 100 samples of 3 tokens.
    single = collect(0)
    assert len(single) == 100
    assert collect(2) == single


def test_in_memory_hf_dataset_can_shuffle(tok, tmp_path):
    files = write_text_files(str(tmp_path))
    cache_dir = str(tmp_path / "cache")

    ordered = as_lists(make_hf_dataset(tok, files, False, cache_dir))
    shuffled = as_lists(
        make_hf_dataset(tok, files, False, cache_dir, shuffle=True, seed=0)
    )

    assert sorted(shuffled) == sorted(ordered)
    assert shuffled != ordered

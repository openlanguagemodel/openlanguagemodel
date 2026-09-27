from typing import List, Optional, Union
from olm.data.tokenization.base import TokenizerBase
import torch
import os


def _import_auto_tokenizer():
    """
    Import ``transformers.AutoTokenizer`` on first use.

    ``import olm`` loads this module, and OLM model code does not need
    ``transformers``, so importing it lazily keeps a missing or broken install
    from breaking ``import olm`` and saves its import time.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "HFTokenizer needs the Hugging Face `transformers` package, which could "
            f"not be imported ({type(exc).__name__}: {exc}). Install or repair it "
            "with `pip install -U transformers`."
        ) from exc
    return AutoTokenizer


class HFTokenizer(TokenizerBase):
    """
    Wrapper around a Hugging Face fast tokenizer.

    ``HFTokenizer`` loads any tokenizer that ``transformers.AutoTokenizer`` can
    load, from the Hub or from a local directory, and exposes the small
    interface OLM uses: ``encode``, ``decode``, ``vocab_size`` and the
    special-token ids. The wrapped tokenizer is available as ``tokenizer`` for
    everything else, such as chat templates or batch encoding.

    ``encode`` returns a 1-D ``torch.long`` tensor by default. Pass
    ``return_tensors=None`` to get a plain ``list[int]`` instead.

    Implementation note:
        ``encode`` builds its tensor with ``torch.tensor`` rather than asking
        ``transformers`` for ``return_tensors="pt"``. Recent ``transformers``
        releases switch off their PyTorch integration when the installed torch
        is older than they support (``transformers`` 5.12 needs torch 2.4 or
        newer), and ``return_tensors="pt"`` then raises ``ImportError`` even
        though torch is installed. Tokenization itself needs no PyTorch support
        from ``transformers``.

    Args:
        model_path (str): Hub model id, such as ``"gpt2"``, or a local directory
            written by ``save_pretrained`` or ``HFTokenizer.save``.

    Attributes:
        tokenizer: The wrapped ``transformers`` tokenizer.
        vocab_size (int): Number of token ids the tokenizer can produce,
            including added special tokens (``len(tokenizer)``). Size embedding
            and output layers with this value.
    """

    def __init__(self, model_path: str):
        AutoTokenizer = _import_auto_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # ``tokenizer.vocab_size`` leaves out added tokens, whose ids come after
        # the base vocabulary, so an embedding sized with it can be too small.
        self.vocab_size = len(self.tokenizer)

    @property
    def eos_token_id(self) -> Optional[int]:
        """Id of the end-of-sequence token, or ``None`` if the tokenizer has none."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Id of the beginning-of-sequence token, or ``None`` if there is none."""
        return self.tokenizer.bos_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Union[torch.Tensor, List[int]]:
        """
        Encode a single string into token ids.

        The result has no batch dimension. Call ``.unsqueeze(0)`` on the tensor
        before passing it to a model. Text longer than the tokenizer's maximum
        length is not truncated.

        Args:
            text (str): The text to tokenize.
            add_special_tokens (bool): Whether to add the tokenizer's own special
                tokens, such as a leading BOS token. The GPT-2 tokenizer adds
                none. OLM datasets pass ``False``.
            return_tensors (Optional[str]): ``"pt"`` (the default) returns a
                ``torch.Tensor``. ``None`` returns a ``list[int]``.

        Returns:
            A 1-D ``torch.long`` tensor of shape ``[num_tokens]`` when
            ``return_tensors="pt"``, or a ``list[int]`` when
            ``return_tensors=None``.

        Raises:
            ValueError: If ``return_tensors`` is neither ``"pt"`` nor ``None``.
        """
        if return_tensors not in ("pt", None):
            raise ValueError(
                f"return_tensors must be 'pt' or None, got {return_tensors!r}."
            )

        input_ids = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )["input_ids"]
        if return_tensors is None:
            return input_ids
        return torch.tensor(input_ids, dtype=torch.long)

    def decode(
        self,
        tokens: Union[torch.Tensor, List[int], int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode one sequence of token ids back into a string.

        Args:
            tokens: A 1-D tensor, a batch of one such as ``[1, num_tokens]``, a
                ``list[int]``, or a single id.
            skip_special_tokens (bool): Whether to drop special tokens such as
                EOS from the text. Detect EOS by comparing ids with
                ``eos_token_id``, not by looking for it in the decoded text.

        Returns:
            str: The decoded text.

        Raises:
            ValueError: If ``tokens`` is a tensor holding more than one sequence,
                such as a ``[batch, num_tokens]`` batch or a ``[batch, 1]``
                column of sampled ids.
        """
        if isinstance(tokens, torch.Tensor):
            # Only leading dimensions of size 1 may be dropped: ``[B, 1]`` holds
            # B sequences of one token, not one sequence of B tokens.
            if tokens.dim() > 1 and tokens.numel() != tokens.shape[-1]:
                raise ValueError(
                    "decode() takes one sequence of token ids, got a tensor of "
                    f"shape {tuple(tokens.shape)}. Decode each row separately, "
                    "for example [tok.decode(row) for row in batch]."
                )
            tokens = tokens.reshape(-1).tolist()
        if isinstance(tokens, int):
            tokens = [tokens]

        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        """
        Saves tokenizer in HuggingFace format.
        `path` must be a directory.
        """
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "type"), "w") as f:
            f.write("HFTokenizer")

    @classmethod
    def load(cls, path: str) -> "HFTokenizer":
        """Load a tokenizer written by ``HFTokenizer.save``."""
        obj = cls.__new__(cls)
        obj.tokenizer = _import_auto_tokenizer().from_pretrained(path, use_fast=True)
        obj.vocab_size = len(obj.tokenizer)
        return obj

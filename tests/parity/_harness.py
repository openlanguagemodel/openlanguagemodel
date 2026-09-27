"""Runner that measures OLM-vs-reference agreement for one architecture.

Everything here is CPU-only, offline, and built from random weights at a tiny
scale, so a full sweep runs in seconds and needs no downloaded checkpoints.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, asdict
from typing import Sequence

import torch
import torch.nn.functional as F

from ._spec import MapEntry, ParityCase

SEED = 0
BATCH = 2
SEQ_LEN = 16

#: Multiplier applied to the reference's matrix-shaped parameters before the
#: comparison. At the shipped init (``initializer_range=0.02``, hidden size 64)
#: the largest logit is around 1, and every non-linearity this suite advertises
#: is then operating in a regime where breaking it barely moves the output:
#: removing Gemma 2's attention soft-cap changes the logits by 2.7e-7, which is
#: indistinguishable from float32 rounding. Scaling the weights moves the model
#: into the non-linear regime, so "this feature is covered" becomes measurable.
#:
#: 5.0 is a measured choice, not a guess. At x5 every matching case still agrees
#: to float32 rounding while the stock-GELU and soft-cap mutations are caught.
#: At x10 some *correct* cases exceed an absolute 1e-5, and matching ``gpt2``
#: fails the per-tensor gradient check on ``k_proj.bias``: that tensor's true
#: gradient is exactly zero, so at x10 it rises out of DEGENERATE_GRAD_RATIO's
#: cutoff and its rounding-noise direction starts being compared.
WEIGHT_SCALE = 5.0


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def set_determinism(seed: int = SEED) -> None:
    """Pin every source of run-to-run variation we can reach."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def environment() -> dict[str, str]:
    """Versions and hardware, recorded alongside every measurement."""
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
    }


# --------------------------------------------------------------------------
# Weight transfer
# --------------------------------------------------------------------------


def _named(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    # remove_duplicate=True collapses tied embedding/output-head parameters to
    # a single entry, which is what we want: they are one tensor.
    return dict(model.named_parameters(remove_duplicate=True))


def randomize_degenerate_parameters(
    model: torch.nn.Module, seed: int, spread: float = 0.1
) -> list[str]:
    """Break up constant-valued reference parameters before comparing.

    ``transformers`` initialises every RMSNorm/LayerNorm scale to exactly ones
    (or, for Gemma, zeros) and every bias to zeros. A constant vector is
    invariant under permutation, so a mapping that permutes such a parameter
    incorrectly -- or forgets to permute it -- still produces identical logits.
    Whole classes of mapping bug are therefore invisible to a forward-only
    comparison against a freshly initialised reference.

    This was not hypothetical: Qwen3's per-head QK-norm scales are exactly the
    parameters that need the RoPE permutation, and the omission only showed up
    once they held non-constant values.

    Each constant parameter is jittered around its original value, keeping the
    model numerically sane while making it order-sensitive.

    Returns:
        The names of the parameters that were randomized.
    """
    generator = torch.Generator().manual_seed(seed)
    touched = []
    with torch.no_grad():
        for name, param in model.named_parameters(remove_duplicate=True):
            if param.numel() < 2:
                continue
            flat = param.reshape(-1)
            if not bool(torch.all(flat == flat[0])):
                continue
            noise = torch.randn(
                param.shape, generator=generator, dtype=torch.float32
            ).to(param.dtype)
            param.copy_(param + noise * spread)
            touched.append(name)
    return touched


def scale_reference_weights(
    model: torch.nn.Module, factor: float = WEIGHT_SCALE, min_dim: int = 2
) -> list[str]:
    """Scale up the reference's matrix parameters, leaving its vectors alone.

    Only parameters with at least ``min_dim`` axes are touched, i.e. projection
    and embedding matrices but not norm scales or biases. Norm scales sit in
    front of a normalisation, so scaling them would partly cancel; biases add a
    constant offset rather than amplifying the signal. Restricting the change to
    matrices is what makes the resulting logit growth roughly geometric in depth
    and keeps every case's weights a plain multiple of what ``transformers``
    initialised.

    ``remove_duplicate=True`` matters: with a tied embedding/output head the two
    names share one tensor, and iterating the duplicates would scale it twice.

    Returns:
        The names of the parameters that were scaled.
    """
    if factor == 1.0:
        return []
    scaled = []
    with torch.no_grad():
        for name, param in model.named_parameters(remove_duplicate=True):
            if param.dim() < min_dim:
                continue
            param.mul_(factor)
            scaled.append(name)
    return scaled


def check_coverage(
    case: ParityCase,
    olm: torch.nn.Module,
    hf: torch.nn.Module,
    entries: Sequence[MapEntry],
) -> None:
    """Fail loudly if the mapping silently skips a parameter on either side.

    Without this, an unmapped OLM parameter would keep its random init and the
    parity number would be meaningless — or worse, an unmapped *reference*
    parameter would mean we are comparing against a different model than we
    think we are.
    """
    olm_params, hf_params = _named(olm), _named(hf)

    mapped_olm = {e.olm for e in entries}
    mapped_hf = {n for e in entries for n in e.hf_names}

    missing_olm = set(olm_params) - mapped_olm - set(case.olm_only)
    missing_hf = set(hf_params) - mapped_hf - set(case.hf_only)
    unknown_olm = mapped_olm - set(olm_params)
    unknown_hf = mapped_hf - set(hf_params)

    problems = []
    if missing_olm:
        problems.append(f"OLM parameters with no mapping: {sorted(missing_olm)}")
    if missing_hf:
        problems.append(f"reference parameters never consumed: {sorted(missing_hf)}")
    if unknown_olm:
        problems.append(f"mapping names nonexistent OLM params: {sorted(unknown_olm)}")
    if unknown_hf:
        problems.append(f"mapping names nonexistent HF params: {sorted(unknown_hf)}")
    if problems:
        raise AssertionError(f"[{case.name}] incomplete weight map:\n  " + "\n  ".join(problems))


def load_reference_weights(
    olm: torch.nn.Module, hf: torch.nn.Module, entries: Sequence[MapEntry]
) -> None:
    """Copy HF reference weights into the OLM model through the explicit map."""
    olm_params, hf_params = _named(olm), _named(hf)

    with torch.no_grad():
        for entry in entries:
            sources = [hf_params[n].detach() for n in entry.hf_names]
            value = entry.apply(*sources)
            target = olm_params[entry.olm]
            if value.shape != target.shape:
                raise AssertionError(
                    f"{entry.olm}: mapped shape {tuple(value.shape)} != "
                    f"OLM shape {tuple(target.shape)} (from {entry.hf_names})"
                )
            target.copy_(value)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross entropy, computed identically for both models.

    Deliberately not delegated to the HF model's own loss path: we want the
    loss difference to reflect the logit difference and nothing else.
    """
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)


#: A parameter whose reference gradient norm is below this fraction of the
#: largest reference gradient norm has no signal to compare: its "direction" is
#: rounding noise. The clearest example is an attention key-projection bias,
#: whose true gradient is exactly zero because softmax is invariant to adding a
#: constant vector to every key. Such tensors are excluded from the per-tensor
#: cosine statistic (they still contribute, negligibly, to the global one).
DEGENERATE_GRAD_RATIO = 1e-8


def _grad_vectors(
    olm: torch.nn.Module, hf: torch.nn.Module, entries: Sequence[MapEntry]
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, float]], list[str]]:
    """Pair gradients through the same linear map used for the weights."""
    olm_params, hf_params = _named(olm), _named(hf)

    olm_chunks, hf_chunks, paired = [], [], []
    for entry in entries:
        olm_grad = olm_params[entry.olm].grad
        hf_grads = [hf_params[n].grad for n in entry.hf_names]
        if olm_grad is None or any(g is None for g in hf_grads):
            continue
        mapped = entry.apply_grad(*hf_grads)
        a, b = olm_grad.reshape(-1).double(), mapped.reshape(-1).double()
        olm_chunks.append(a)
        hf_chunks.append(b)
        paired.append((entry.olm, a, b))

    reference_scale = max((float(b.norm()) for _, _, b in paired), default=0.0)
    cutoff = reference_scale * DEGENERATE_GRAD_RATIO

    per_tensor, degenerate = [], []
    for name, a, b in paired:
        if float(b.norm()) <= cutoff:
            degenerate.append(name)
            continue
        denom = a.norm() * b.norm()
        if denom > 0:
            per_tensor.append((name, float(torch.dot(a, b) / denom)))

    return torch.cat(olm_chunks), torch.cat(hf_chunks), per_tensor, degenerate


@dataclass
class ParityResult:
    name: str
    reference: str
    dtype: str
    max_abs_logit_diff: float
    max_abs_loss_diff: float
    grad_cosine_sim: float
    min_per_tensor_grad_cosine: float
    worst_grad_tensor: str
    logit_scale: float
    loss_scale: float
    n_params: int
    n_degenerate_grads: int = 0
    reference_deviations: tuple[str, ...] = ()
    weight_scale: float = WEIGHT_SCALE

    @property
    def rel_logit_diff(self) -> float:
        """Logit disagreement as a fraction of the reference's logit scale.

        The absolute difference is meaningless on its own: it grows with the
        init scale, the config and the platform. Divided by ``max |logit|`` it
        is a statement about rounding, which is what the claim actually is.
        """
        return self.max_abs_logit_diff / self.logit_scale

    @property
    def rel_loss_diff(self) -> float:
        """Loss disagreement as a fraction of the reference loss."""
        return self.max_abs_loss_diff / self.loss_scale

    def as_dict(self) -> dict:
        data = asdict(self)
        data["rel_logit_diff"] = self.rel_logit_diff
        data["rel_loss_diff"] = self.rel_loss_diff
        return data


def run_parity(
    case: ParityCase,
    dtype: torch.dtype = torch.float64,
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    seed: int = SEED,
    weight_scale: float = WEIGHT_SCALE,
) -> ParityResult:
    """Build both models from one config, share weights, and measure agreement."""
    set_determinism(seed)

    hf = case.build_hf().to(dtype).eval()
    olm = case.build_olm().to(dtype).eval()

    randomize_degenerate_parameters(hf, seed=seed + 2)
    # After the jitter, so the scale applies to the values actually compared,
    # and before the map, so OLM receives exactly these weights.
    scale_reference_weights(hf, factor=weight_scale)

    entries = list(case.build_map(olm, hf))
    check_coverage(case, olm, hf, entries)
    load_reference_weights(olm, hf, entries)

    generator = torch.Generator().manual_seed(seed + 1)
    vocab = hf.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq_len), generator=generator)

    olm_logits = olm(input_ids)
    hf_logits = hf(input_ids).logits.to(dtype)

    if olm_logits.shape != hf_logits.shape:
        raise AssertionError(
            f"[{case.name}] logit shape mismatch: "
            f"OLM {tuple(olm_logits.shape)} vs reference {tuple(hf_logits.shape)}"
        )

    max_logit_diff = float((olm_logits - hf_logits).abs().max())

    olm_loss = causal_lm_loss(olm_logits, input_ids)
    hf_loss = causal_lm_loss(hf_logits, input_ids)
    max_loss_diff = float((olm_loss - hf_loss).abs())

    olm.zero_grad(set_to_none=True)
    hf.zero_grad(set_to_none=True)
    olm_loss.backward()
    hf_loss.backward()

    olm_vec, hf_vec, per_tensor, degenerate = _grad_vectors(olm, hf, entries)
    denom = olm_vec.norm() * hf_vec.norm()
    global_cos = float(torch.dot(olm_vec, hf_vec) / denom) if denom > 0 else float("nan")
    worst_name, worst_cos = (
        min(per_tensor, key=lambda kv: kv[1]) if per_tensor else ("<none>", float("nan"))
    )

    return ParityResult(
        name=case.name,
        reference=case.reference,
        dtype=str(dtype).replace("torch.", ""),
        max_abs_logit_diff=max_logit_diff,
        max_abs_loss_diff=max_loss_diff,
        grad_cosine_sim=global_cos,
        min_per_tensor_grad_cosine=worst_cos,
        worst_grad_tensor=worst_name,
        logit_scale=float(hf_logits.abs().max()),
        loss_scale=float(hf_loss.abs()),
        n_params=sum(p.numel() for p in _named(olm).values()),
        n_degenerate_grads=len(degenerate),
        reference_deviations=case.reference_deviations,
        weight_scale=weight_scale,
    )

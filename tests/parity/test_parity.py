"""Reference-parity tests: OLM architectures vs their ``transformers`` twins.

Run with::

    pytest tests/parity -q

Everything is CPU-only, offline, and randomly initialised at a tiny scale, with
the reference's matrix weights scaled up so that the non-linear features under
test actually affect the output (see ``_harness.WEIGHT_SCALE``).
"""

from __future__ import annotations

import pytest
import torch

from ._harness import (
    WEIGHT_SCALE,
    causal_lm_loss,
    check_coverage,
    load_reference_weights,
    randomize_degenerate_parameters,
    run_parity,
    scale_reference_weights,
    set_determinism,
)
from ._spec import ParityCase
from .architectures import ALL_CASES, CASES_BY_NAME
from .architectures import gemma2 as gemma2_module
from .architectures import llama as llama_module

#: Cases whose reference is deliberately left at a default that OLM does not
#: implement. They are asserted to *disagree*, which doubles as a check that the
#: harness is capable of detecting a real discrepancy.
KNOWN_DIVERGENT = {"gpt2-stock", "gemma2-stock"}

MATCHING_CASES = [c for c in ALL_CASES if c.name not in KNOWN_DIVERGENT]
DIVERGENT_CASES = [c for c in ALL_CASES if c.name in KNOWN_DIVERGENT]

DTYPES = [torch.float64, torch.float32]

#: Tolerances are **relative**, and deliberately not tuned per architecture.
#: An absolute logit bound is not a statement about anything durable: the same
#: correct code measures a different absolute difference when the init scale,
#: the config or the platform changes, because what is being measured is
#: float32 rounding at the size of the logits. Dividing by the reference's own
#: ``max |logit|`` removes exactly that dependence.
#:
#: OLM's LayerNorm and RMSNorm both upcast to float32 internally and cast back,
#: so float64 runs cannot resolve below roughly float32 epsilon either; both
#: dtypes therefore share one tolerance.
#:
#: 1e-5 is where the measurements put it: across all 11 matching cases and both
#: dtypes the worst relative logit difference is 6.1e-7, and the weakest of the
#: feature mutations in ``FEATURE_MUTATIONS`` below is 1.6e-4. The bound sits
#: about 16x above every correct case and 16x below every broken one.
REL_LOGIT_TOL = 1e-5

#: The loss is a mean over ``batch * (seq_len - 1)`` positions, so it averages
#: logit disagreement away and is the less sensitive of the two; it is here to
#: catch a loss path that differs by more than rounding, not to detect features.
#: 1e-6 relative sits between the worst matching case (1.1e-7) and the smallest
#: real divergence the suite contains (``gemma2-stock``, 9.7e-6). In absolute
#: terms it is about eight float32 ulps of a loss near ln 61 = 4.1; the previous
#: absolute 1e-6 was two ulps, and ``gpt2`` already measured one.
REL_LOSS_TOL = 1e-6

GRAD_COSINE_TOL = 1e-9


def _ids(case: ParityCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_weight_map_is_complete(case: ParityCase) -> None:
    """Every parameter on both sides must be named in the mapping.

    This is what stops a parity number from being accidentally meaningless: an
    unmapped OLM parameter would silently keep its random initialisation.
    """
    torch.manual_seed(0)
    hf = case.build_hf()
    olm = case.build_olm()
    check_coverage(case, olm, hf, list(case.build_map(olm, hf)))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("case", MATCHING_CASES, ids=_ids)
def test_matches_reference(case: ParityCase, dtype: torch.dtype) -> None:
    result = run_parity(case, dtype=dtype)

    assert result.rel_logit_diff < REL_LOGIT_TOL, (
        f"{case.name}/{result.dtype}: max |logit difference| / max |logit| "
        f"{result.rel_logit_diff:.3e} exceeds {REL_LOGIT_TOL:.0e} "
        f"({result.max_abs_logit_diff:.3e} absolute, reference logit scale "
        f"{result.logit_scale:.3f}, reference weights x{result.weight_scale:g})"
    )
    assert result.rel_loss_diff < REL_LOSS_TOL, (
        f"{case.name}/{result.dtype}: max |loss difference| / loss "
        f"{result.rel_loss_diff:.3e} exceeds {REL_LOSS_TOL:.0e} "
        f"({result.max_abs_loss_diff:.3e} absolute on a loss of "
        f"{result.loss_scale:.4f})"
    )
    assert result.grad_cosine_sim > 1.0 - GRAD_COSINE_TOL, (
        f"{case.name}/{result.dtype}: gradient cosine similarity "
        f"{result.grad_cosine_sim:.12f} below 1 - {GRAD_COSINE_TOL:.0e}"
    )
    assert result.min_per_tensor_grad_cosine > 1.0 - 1e-6, (
        f"{case.name}/{result.dtype}: worst per-parameter gradient cosine "
        f"{result.min_per_tensor_grad_cosine:.9f} on {result.worst_grad_tensor}"
    )


#: Each divergent case and the matched-activation case it is compared against.
DIVERGENCE_PAIRS = {"gpt2-stock": "gpt2", "gemma2-stock": "gemma2"}

#: How much larger the stock-reference disagreement must be than the
#: matched-reference one before we call it a real difference rather than noise.
DIVERGENCE_RATIO = 10.0


@pytest.mark.parametrize("case", DIVERGENT_CASES, ids=_ids)
def test_known_divergence_is_detected(case: ParityCase) -> None:
    """OLM does not reproduce these references, and the harness must say so.

    OLM's ClassicFFN and GeGLU both use exact erf GELU, while stock GPT-2 uses
    ``gelu_new`` and stock Gemma 2 uses ``gelu_pytorch_tanh``. The divergence is
    small in absolute terms, so it is judged against the *same architecture with
    the activation aligned* rather than against a fixed tolerance: that isolates
    the activation as the cause and keeps the finding from quietly regressing
    into a pass.
    """
    baseline_name = DIVERGENCE_PAIRS[case.name]
    baseline = run_parity(CASES_BY_NAME[baseline_name], dtype=torch.float64)
    result = run_parity(case, dtype=torch.float64)

    ratio = result.max_abs_logit_diff / max(baseline.max_abs_logit_diff, 1e-300)
    assert ratio > DIVERGENCE_RATIO, (
        f"{case.name} matched its stock reference about as well as "
        f"{baseline_name} matched its activation-aligned one "
        f"({result.max_abs_logit_diff:.3e} vs {baseline.max_abs_logit_diff:.3e}); "
        "the documented activation-function difference may have been fixed, in "
        "which case move this case out of KNOWN_DIVERGENT."
    )


def test_harness_detects_a_broken_mapping() -> None:
    """Negative control: corrupting the mapping must break parity.

    Without this, a harness that accidentally compared a model to itself would
    report perfect agreement forever. Here the RoPE head permutation is dropped
    from Llama 3's Q/K projections -- the single subtlest part of the whole
    mapping -- and parity must collapse.
    """
    import dataclasses

    from ._spec import MapEntry

    # Selected by name, not by index: the index shifts whenever a case is added
    # to llama.py, and silently running this negative control against a
    # different architecture than the docstring claims would defeat its purpose.
    case = next(c for c in llama_module.CASES if c.name == "llama3")

    def broken_map(olm, hf) -> list[MapEntry]:
        return [
            dataclasses.replace(e, fn=None, grad_fn=None)
            if (".q_proj.weight" in e.olm or ".k_proj.weight" in e.olm)
            else e
            for e in case.build_map(olm, hf)
        ]

    broken = dataclasses.replace(case, name="llama3-broken", build_map=broken_map)
    result = run_parity(broken, dtype=torch.float64)

    assert result.max_abs_logit_diff > 1e-3, (
        "dropping the RoPE convention permutation did not change the logits; "
        "the harness is not measuring what it claims to measure"
    )


# --------------------------------------------------------------------------
# Per-feature negative controls
# --------------------------------------------------------------------------


def _gemma2_with(name: str, **overrides) -> ParityCase:
    """Gemma 2, with exactly one OLM feature knob changed and nothing else."""
    import dataclasses

    return dataclasses.replace(
        CASES_BY_NAME["gemma2"],
        name=f"gemma2[{name}]",
        build_olm=lambda: gemma2_module.build_olm(**overrides),
    )


def _qwen3_moe_without_qk_norm_permutation() -> ParityCase:
    """Qwen3-MoE with the per-head QK-norm scales left in reference order."""
    import dataclasses

    from ._spec import MapEntry

    case = CASES_BY_NAME["qwen3-moe"]

    def broken_map(olm, hf) -> list[MapEntry]:
        return [
            (
                dataclasses.replace(e, fn=None, grad_fn=None)
                if (".q_norm.weight" in e.olm or ".k_norm.weight" in e.olm)
                else e
            )
            for e in case.build_map(olm, hf)
        ]

    return dataclasses.replace(
        case, name="qwen3-moe[qk-norm-unpermuted]", build_map=broken_map
    )


#: One mutation per feature the README lists as covered, with the relative logit
#: difference measured on transformers 4.57.1 / torch 2.2.2 at the default
#: ``WEIGHT_SCALE``. Each breaks a single feature and nothing else, so
#: ``test_matches_reference`` must reject it. Without this, "Gemma 2's soft-caps
#: are covered" means only that a case exists with soft-capping switched on --
#: not that the comparison can see it. It could not: at the shipped init scale
#: the attention soft-cap mutation measures 6.5e-7 relative, i.e. *passes*.
FEATURE_MUTATIONS = [
    (
        "attention-logit-softcap",
        _gemma2_with("no-attn-softcap", attn_logit_softcap=None),
        1.6e-4,
    ),
    (
        "final-logit-softcap",
        _gemma2_with("no-final-softcap", final_logit_softcap=None),
        5.0e-3,
    ),
    (
        "query-pre-attn-scalar",
        _gemma2_with("default-query-scale", query_pre_attn_scalar=None),
        8.7e-2,
    ),
    ("sliding-window", _gemma2_with("window-off-by-one", sliding_window=9), 2.7e-1),
    ("gelu-variant", CASES_BY_NAME["gemma2-stock"], 2.2e-4),
    ("qk-norm-permutation", _qwen3_moe_without_qk_norm_permutation(), 3.2e-1),
]


@pytest.mark.parametrize(
    "feature,case,measured",
    FEATURE_MUTATIONS,
    ids=[m[0] for m in FEATURE_MUTATIONS],
)
def test_advertised_feature_is_detectably_covered(
    feature: str, case: ParityCase, measured: float
) -> None:
    """Breaking one advertised feature must fail the parity criterion.

    Asserted against ``REL_LOGIT_TOL`` itself rather than a separate threshold,
    so the two tests cannot drift apart: loosening the tolerance until a real
    difference slips through fails here immediately.

    float64 only. Every mutation below is at least 16x the tolerance and more
    than two orders of magnitude above float32 rounding, so the dtype makes no
    difference (checked: the two agree to three significant figures), and one
    dtype keeps the suite at a few seconds.
    """
    result = run_parity(case, dtype=torch.float64)

    assert result.rel_logit_diff > REL_LOGIT_TOL, (
        f"breaking {feature} ({case.name}) left max |logit difference| / "
        f"max |logit| at {result.rel_logit_diff:.3e}, within the "
        f"{REL_LOGIT_TOL:.0e} the matching cases are held to, so the suite "
        f"would not notice if OLM got this feature wrong. Measured "
        f"{measured:.1e} when written; if OLM's behaviour changed "
        "deliberately, update the case, not this bound."
    )


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_reference_constants_are_randomized(case: ParityCase) -> None:
    """Norm scales and biases must not still be constant when we compare.

    ``transformers`` initialises them to exactly ones or zeros, and constant
    vectors are permutation-invariant, so leaving them alone would hide any
    mapping bug affecting them -- which is exactly how Qwen3's QK-norm
    permutation went unnoticed until the gradient check caught it.
    """
    torch.manual_seed(0)
    hf = case.build_hf()
    randomize_degenerate_parameters(hf, seed=2)

    still_constant = [
        name
        for name, p in hf.named_parameters(remove_duplicate=True)
        if p.numel() > 1 and bool(torch.all(p.reshape(-1) == p.reshape(-1)[0]))
    ]
    assert not still_constant, (
        f"{case.name}: parameters left at a constant value, so a permutation "
        f"error in them would be undetectable: {still_constant}"
    )


def test_weight_scaling_touches_matrices_once_and_leaves_vectors_alone() -> None:
    """Guard the scaling itself, including the tied-parameter trap.

    With a tied embedding and output head the two names are one tensor, so
    iterating duplicates would scale it by ``WEIGHT_SCALE ** 2`` and quietly
    compare against a model that is not a multiple of the reference. Norm scales
    and biases must be left alone: they are the parameters
    ``randomize_degenerate_parameters`` has just jittered around 1 and 0, and
    amplifying those is not the same experiment.
    """
    case = CASES_BY_NAME["llama2-mha-tied"]
    torch.manual_seed(0)
    hf = case.build_hf()
    before = {n: p.detach().clone() for n, p in hf.named_parameters()}

    scaled = scale_reference_weights(hf, factor=WEIGHT_SCALE)

    assert "model.embed_tokens.weight" in scaled
    for name, param in hf.named_parameters():
        expected = before[name] * (WEIGHT_SCALE if param.dim() >= 2 else 1.0)
        assert torch.equal(param.detach(), expected), (
            f"{name} (dim {param.dim()}) was not scaled by exactly "
            f"{'WEIGHT_SCALE' if param.dim() >= 2 else '1'}"
        )


def test_loss_is_computed_identically_for_both_models() -> None:
    """The loss difference must reflect the logit difference and nothing else."""
    logits = torch.randn(2, 8, 11, dtype=torch.float64)
    labels = torch.randint(0, 11, (2, 8))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 11), labels[:, 1:].reshape(-1)
    )
    assert torch.equal(causal_lm_loss(logits, labels), expected)


def test_reference_weights_actually_land_in_the_olm_model() -> None:
    """Guard the copy itself: a no-op ``load`` would fake perfect parity."""
    case = MATCHING_CASES[0]
    torch.manual_seed(0)
    hf = case.build_hf().to(torch.float64)
    olm = case.build_olm().to(torch.float64)
    entries = list(case.build_map(olm, hf))

    olm_params = dict(olm.named_parameters(remove_duplicate=True))
    entry = entries[0]
    before = olm_params[entry.olm].detach().clone()

    load_reference_weights(olm, hf, entries)
    after = olm_params[entry.olm].detach()

    hf_params = dict(hf.named_parameters(remove_duplicate=True))
    expected = entry.apply(*[hf_params[n].detach() for n in entry.hf_names])
    assert torch.equal(after, expected)
    assert not torch.equal(after, before)


def test_suite_does_not_leak_global_torch_settings() -> None:
    """The parity fixtures must not change torch's global state for later tests.

    ``set_determinism`` flips three process-global switches. If they are not put
    back, every test that runs after this package in a full session inherits
    deterministic algorithms and disabled cuDNN autotuning, which silently
    changes kernel selection and performance elsewhere. This asserts the
    restore covers all three, not just the thread count.
    """
    from .conftest import preserved_torch_globals

    def snapshot():
        return (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark,
            torch.get_num_threads(),
        )

    # Start from the opposite of what the suite sets, so a restore that merely
    # hardcoded torch's defaults would fail here too.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    baseline = snapshot()

    with preserved_torch_globals():
        set_determinism()
        assert snapshot() != baseline, "set_determinism did not change anything"

    assert snapshot() == baseline

# Reference-parity harness

Checks OLM architectures numerically against their HuggingFace `transformers`
reference implementations, and reports the measured disagreement.

```bash
pytest tests/parity -q                                  # run the suite
PYTHONPATH=src:tests python -m parity.report            # regenerate the table
```

The whole suite is CPU-only, offline, and finishes in a few seconds. It does not
download checkpoints.

**The numbers are not committed.** `python -m parity.report` regenerates them,
and records the torch/transformers versions and hardware alongside them, because
that is what they depend on: every value in the table is a small multiple of the
float32 ulp at the size of the logits, so it moves with the platform, the config
and the init scale. A committed table would go stale silently and read as a
property of OLM. [RESULTS.md](RESULTS.md) keeps only the part that is *not* a
measurement: the list of places where the reference config is deliberately not
its default.

What the suite asserts, and the form any claim from it should take:

> OLM matches `transformers` to float32 rounding — max |Δlogit| / max |logit|
> below 1e-5 — on tiny random-init configs at `WEIGHT_SCALE`, for the
> architectures listed under Scope.

The bound is relative on purpose; see [Tolerances](#tolerances).

## Environment

Needs `transformers >= 4.57.1` (for the Qwen3 and OLMo references) and a numpy
that matches the installed torch. The suite handles both reference layouts, 4.x
and 5.x, so either line works. The common failure — `Failed to initialize
NumPy: _ARRAY_API not found` — is numpy 2.x against a torch built against numpy
1.x; pin `numpy<2` for torch 2.2.x.

```bash
uv venv --python 3.12 .venv-parity
VIRTUAL_ENV=$PWD/.venv-parity uv pip install \
    torch 'numpy<2' 'transformers==4.57.1' pytest datasets tqdm pyyaml safetensors
VIRTUAL_ENV=$PWD/.venv-parity PYTHONPATH=$PWD/src .venv-parity/bin/python -m pytest tests/parity -q
```

`transformers` is pinned rather than lower-bounded in that recipe because on a
machine capped at torch 2.2.2 an unbounded `>=4.57.1` resolves to 5.x, which
requires torch >= 2.5 and then reports `PyTorch was not found` and offers no
models at all. On a current torch, drop the pin.

`datasets` is required only because `import olm` pulls in `olm.data.datasets`.

> On macOS x86_64 the newest installable torch is **2.2.2** — PyTorch stopped
> publishing Intel-Mac wheels after that release, which also caps the installable
> `transformers` at the 4.x line. Nothing in this harness requires a newer torch,
> but a Linux/arm64 machine can and should re-run it on current torch; the
> reported numbers are hardware- and version-specific.

## Scope

Covered, matching their reference: **GPT-2, Llama 2 (MHA and GQA), Llama 3,
Qwen 2.5, Mistral, Phi-3, Gemma 2, OLMo, Qwen3-MoE.**

Not covered, because `transformers` ships no reference implementation for them
at any scale — including all six OLM architectures that currently have no test
coverage at all:

| OLM model | why not |
|---|---|
| `inclusionai/ling2_5` | no HF architecture |
| `minimax/m2_5` | no HF architecture (`MiniMaxForCausalLM` is MiniMax-Text-01, a different model) |
| `nanbeige/nanbeige4_1` | no HF architecture |
| `alibaba/qwen3_5` | no HF architecture |
| `sarvam/sarvam` | no HF architecture |
| `stepfun/step3_5` | no HF architecture |

For these, parity against a reference is not possible; they need conventional
unit tests instead, and this harness does not pretend otherwise.

Coverable but not yet done — a reference exists in `transformers 4.57.1`, so
these are the natural next additions: `gemma3`, `llama4`, `deepseek_v3`,
`qwen3_next`, `olmo3`, `opt`, `phi4`, `mistral_large3`.

## What it measures

For each architecture, three quantities, computed on the same inputs with the
same weights:

| quantity | meaning |
|---|---|
| max abs Δlogit / max abs logit | largest elementwise logit difference, as a fraction of the reference's own logit scale — the reportable one |
| max abs Δloss / loss | difference in next-token cross-entropy, computed identically for both, over the reference loss |
| grad cosine sim | cosine similarity of the full flattened gradient, with parameters paired through the weight map |

The absolute differences are reported too, but only so a reader can see that
they are ulp-sized; they are not the claim.

The gradient is what makes this a test of *training* equivalence and not just
inference equivalence, and it is strictly the most sensitive of the three — see
"Why gradients matter" below.

One limit on that claim, worth stating plainly: the objective compared is
next-token cross entropy and nothing else. Where a reference model adds a second
term to its training loss, that term is outside this suite. The concrete case is
`qwen3-moe`: the router load-balancing auxiliary loss is not compared, because
OLM's `Qwen3Model` discards router logits and so cannot produce it. Load
balancing is exactly what keeps an MoE router from collapsing onto a few
experts, so agreement here does not by itself establish that MoE *training*
matches. Any case with this kind of gap records it in `reference_deviations`, and both
`RESULTS.md` and `python -m parity.report` reprint it.

## How it works

1. Build the `transformers` reference from a tiny config: 2 layers, hidden 64,
   4 heads, vocab 61, sequence 16. Random weights, fixed seed, no checkpoint.
2. **Randomize constant parameters** in the reference (see below).
3. Build the OLM model from the *same* config.
4. Copy reference weights into the OLM model through an explicit, per-parameter
   map — never by matching names.
5. Assert the map covers every parameter on both sides.
6. Compare logits, loss, and gradients in both float32 and float64.

Deliberate choices worth knowing about:

- **Vocab 61 and intermediate 48** are not powers of two, and `head_dim` does
  not equal `hidden_size`, so an incorrect reshape cannot silently pass.
- **`attn_implementation="eager"`** on the reference side, so the comparison is
  against the readable textbook path rather than a fused kernel.
- **Loss is computed by the harness for both models**, not taken from the HF
  model's own loss head, so the loss difference reflects the logit difference
  and nothing else.
- Dropout is zero, both models are in `eval()`, and
  `torch.use_deterministic_algorithms` is on.

### Scaling the reference weights

`transformers` initialises at `initializer_range=0.02`, which at hidden size 64
puts the largest logit near 1 and leaves the models **nearly linear**. Several
of the features this suite claims to cover are then barely reaching the output,
so breaking them does not move the logits. Measured: removing Gemma 2's
attention logit soft-cap on the OLM side changes them by 3.3e-7 relative, which
*passes* the parity test.

`scale_reference_weights` therefore multiplies the reference's ≥2-D parameters
by `WEIGHT_SCALE = 5.0` before the comparison, and OLM receives exactly those
weights. Norm scales and biases are left alone: they are the parameters
`randomize_degenerate_parameters` has just jittered around 1 and 0, and
amplifying those is a different experiment. With a tied embedding and output
head the two names are one tensor, so the scaling must not iterate duplicates —
`test_weight_scaling_touches_matrices_once_and_leaves_vectors_alone` pins that.

5.0 is measured, not picked. At ×5 all 22 matching measurements stay at float32
rounding (worst 6.1e-7 relative) while every feature mutation is caught. At ×10
some *correct* cases exceed an absolute 1e-5, and matching `gpt2` fails the
per-tensor gradient check on `k_proj.bias`: that tensor's true gradient is
exactly zero, so at ×10 it rises out of the degenerate cutoff and its
rounding-noise direction starts being compared.

### Negative controls, one per advertised feature

"Gemma 2's soft-caps are covered" is a claim about the *comparison*, not about
the config, so each advertised feature has a test that breaks exactly that
feature on one side and requires parity to collapse. They assert against
`REL_LOGIT_TOL` itself rather than a threshold of their own, so loosening the
tolerance until a real difference slips through fails them immediately.

Relative logit difference for each mutation, float64, at `WEIGHT_SCALE`
(transformers 4.57.1, torch 2.2.2, CPU):

| broken feature | mutation | max abs Δlogit / max abs logit |
|---|---|---:|
| attention logit soft-cap | OLM `attn_logit_softcap=None` | 1.6e-4 |
| final logit soft-cap | OLM `final_logit_softcap=None` | 5.0e-3 |
| query pre-attention scalar | OLM `query_pre_attn_scalar=None` | 8.8e-2 |
| sliding window | OLM window off by one | 2.8e-1 |
| GELU variant | reference at stock `gelu_pytorch_tanh` | 2.3e-4 |
| QK-norm permutation | per-head scales left in reference order | 3.2e-1 |

Only the first of those needed the weight scaling; the rest were already
visible. The query-scale row needs `query_pre_attn_scalar != head_dim` in the
case config — at `16 == head_dim` the mutation is bit-identical to the
baseline, because it is then the default 1/√head_dim.

### Randomizing constant parameters

`transformers` initialises every norm scale to exactly ones (zeros for Gemma,
which uses `1 + w`) and every bias to zeros. **A constant vector is invariant
under permutation.** Comparing against a freshly initialised reference therefore
cannot detect a mapping bug in any of those parameters, and norm scales are
precisely where per-head permutation bugs live.

`randomize_degenerate_parameters` jitters every constant-valued reference
parameter around its original value before the comparison. This is not
theoretical: Qwen3's per-head QK-norm scales *do* need the RoPE permutation, the
map originally omitted it, and both the logits and the loss matched anyway. Only
the gradient check caught it, and only after randomization made the scales
order-sensitive.

### Why gradients matter

Forward agreement is necessary but not sufficient. A mapping error confined to a
parameter that happens to be constant, or that lands in a subspace the forward
pass is invariant to, produces identical logits and different gradients. Two of
the three real bugs found while building this harness were invisible to the
logit check.

Note that the *per-parameter* cosine is reported alongside the global one: a
single badly mapped tensor barely moves a global cosine dominated by the
embedding matrix.

Parameters whose reference gradient is numerically zero are excluded from the
per-parameter statistic. The clearest example is an attention key-projection
bias: adding a constant vector to every key shifts all attention logits for a
given query by the same amount, softmax is invariant to that, so the true
gradient is exactly zero and its "direction" is pure rounding noise.

## The RoPE convention bridge

The single subtlest part of the mapping. HuggingFace's Llama-family `rotate_half`
pairs head dimension `i` with `i + head_dim/2`. OLM's `RotaryPositionalEmbedding`
pairs adjacent dimensions `2j` and `2j+1`.

These are the same rotation under a permutation of the head axis: OLM slot `2j`
corresponds to HF slot `j`, and OLM slot `2j+1` to HF slot `j + head_dim/2`.
It is the same permutation Meta's official checkpoint-conversion script applies,
in the opposite direction.

`rope_permute` applies it to **Q and K only**. Values are not rotated, and
attention logits are invariant under the *same* permutation applied to both Q
and K, so `v_proj` and `o_proj` stay in the reference layout. Anything that acts
on the head axis between the Q/K projection and RoPE — per-head QK-norm — must be
permuted too.

`test_harness_detects_a_broken_mapping` deletes this permutation from Llama 3 and
asserts that parity collapses, so the suite cannot degenerate into comparing a
model against itself.

## Adding an architecture

Create `architectures/<name>.py` exporting a `CASES` list, and add it to
`architectures/__init__.py`.

```python
from .._spec import MapEntry, ParityCase, cat_rows, rope_permute

def _build_map(olm, hf) -> list[MapEntry]:
    return [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry(
            "blocks.1.stack.0.blocks.0.block.blocks.1.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            rope_permute(num_heads, head_dim),
        ),
        MapEntry(                      # one OLM tensor from several HF tensors
            "blocks.1.stack.0.blocks.1.block.blocks.1.up_proj.weight",
            ("model.layers.0.mlp.up_proj.weight",
             "model.layers.0.mlp.gate_proj.weight"),
            cat_rows,
        ),
    ]

CASES = [ParityCase(name="...", reference="...", build_hf=..., build_olm=...,
                    build_map=_build_map)]
```

Practical notes:

- Print `model.named_parameters()` for both sides first. OLM's paths look
  positional (`blocks.1.stack.0.blocks.0.block.blocks.1...`) because models are
  built from anonymous `Block`/`Repeat`/`Residual` combinators.
- The coverage assertion will tell you exactly what you missed. Genuinely
  one-sided parameters go in `olm_only`/`hf_only` **with a reason**.
- Transforms must be linear. If yours is *affine* — Gemma's `1 + w` norm scale
  is the example — give `MapEntry` a `grad_fn` with the linear part only
  (`identity` for a shift). Applying an affine transform to a gradient is wrong
  and will show up as a wrecked cosine.
- OLM's gated FFNs fuse gate and value into one `up_proj` and split as
  `(value, gate)`. HF Llama keeps them separate as `silu(gate) * up`; HF Phi-3
  fuses them the other way round, as `(gate, up)`. Check which you have.
- Set `pad_token_id=None`. `nn.Embedding(padding_idx=...)` force-zeroes that
  row's gradient in the reference only, and OLM has no equivalent.
- If OLM genuinely does not implement something the reference does by default,
  record it in `reference_deviations` rather than quietly matching the config.
  If OLM cannot match the reference at all, add the case to `KNOWN_DIVERGENT`
  so the disagreement is asserted instead of hidden.

## Tolerances

`REL_LOGIT_TOL = 1e-5` on max |Δlogit| / max |logit|, `REL_LOSS_TOL = 1e-6` on
max |Δloss| / loss, gradient cosine within `1e-9` of 1. Uniform across
architectures, and not tuned per case to make anything pass.

The logit and loss bounds are **relative** because an absolute one is not a
statement about anything durable: the same correct code measures a different
absolute difference when the init scale, the config or the platform changes,
since what is being measured is float32 rounding at the size of the logits.
Dividing by the reference's own `max |logit|` removes exactly that dependence.
1e-5 sits about 16× above the worst of the 22 matching measurements (6.1e-7) and
16× below the weakest feature mutation (1.6e-4).

The loss is a mean over `batch * (seq_len - 1)` positions, so it averages logit
disagreement away and is the less sensitive of the two; it is there to catch a
loss path that differs by more than rounding, not to detect features. 1e-6
relative is about eight float32 ulps of a loss near ln 61 ≈ 4.1, and sits
between the worst matching case (1.1e-7) and the smallest real divergence the
suite contains (`gemma2-stock`, 9.7e-6). An absolute 1e-6 was two ulps, and
`gpt2` already measured one.

Both dtypes share one tolerance because OLM's `LayerNorm` and `RMSNorm` both
upcast to float32 internally and cast back, regardless of input dtype. float64
runs therefore cannot resolve below roughly float32 epsilon either, which is
visible in the results: float64 differences are ~1e-7, not ~1e-15. This is a
reasonable stability choice for low-precision training, but it does mean the
harness cannot currently separate algorithmic differences from float32 rounding
below that floor.

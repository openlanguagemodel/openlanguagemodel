# OLM reference-parity: configuration notes

**The measured numbers are not committed.** Regenerate them with:

    PYTHONPATH=src:tests python -m parity.report          # markdown
    PYTHONPATH=src:tests python -m parity.report --json   # machine-readable

The report records the torch/transformers versions, the platform and the
reference weight scale alongside every value, because that is what the values
depend on: each one is a small multiple of the float32 ulp at the size of the
logits, so it moves with the platform, the config and the init scale. The
assertion that does not move is the relative bound the suite enforces — see
[README.md](README.md#tolerances).

What is kept here is the part that is a *decision* rather than a measurement:
every place where the reference is deliberately not at its own default, or where
the comparison is narrower than the architecture. `python -m parity.report`
reprints this list from `reference_deviations`, so the two cannot disagree.

## Reference configuration deviations

- `gpt2`: activation_function='gelu' (exact erf) instead of the GPT-2 default 'gelu_new'; OLM's ClassicFFN hardcodes nn.GELU(approximate='none')
- `llama2-mha-tied`: tie_word_embeddings=True, whereas released Llama 2/3 checkpoints leave the output head untied.
- `mistral`: head_dim is derived as embed_dim / num_heads. The released Mistral-Small-3.1-24B config sets an explicit head_dim=128 with hidden_size=5120 and 32 query heads, so q_proj is 4096 wide rather than 5120. MistralSmall3_1_Model takes no head_dim argument, so its 24B preset derives 160-dimensional heads and cannot reproduce the released projection shapes. This case therefore validates the block graph, not the advertised preset's geometry. Tracked upstream; compare Qwen3Model, which does accept head_dim.
- `gemma2`: hidden_activation='gelu' (exact erf) instead of the Gemma 2 default 'gelu_pytorch_tanh'; OLM's GeGLU calls F.gelu with no approximation
- `qwen3-moe`: The router load-balancing auxiliary loss is NOT compared. Both sides are scored with plain next-token cross entropy computed from logits, which excludes the auxiliary objective and its router gradients. Qwen3MoeForCausalLM can return router logits and add its configured aux loss, but OLM's Qwen3Model routes through SwiGLUMoEFFN -> MoEFeedForwardBase, whose forward() returns only the output tensor and discards the router logits, so there is nothing to compare against. Read this case as forward and LM-loss parity, not as evidence that real MoE training is equivalent: with no aux loss the two can agree here and still diverge in practice, since load balancing is what stops the router collapsing onto a few experts. OLM's newer olm.nn.moe.MoEFFN does return router logits.

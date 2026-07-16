"""Hugging Face numerical parity integration tests."""

from __future__ import annotations

import pytest
import torch
import yaml

from benchmarks.demo2026.parity import compare, models


FAMILIES = ["gpt2", "llama3", "qwen2"]
CONFIG_DIR = "benchmarks/demo2026/configs/parity"


def _load(family: str) -> dict:
    with open(f"{CONFIG_DIR}/{family}.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parity
@pytest.mark.parametrize("family", FAMILIES)
def test_weight_map_covers_all_parameters(family):
    config = _load(family)
    models.set_determinism(0)
    olm, hf, weight_map, ignored = models.build_pair(
        family, config, device="cpu", init_seed=0
    )
    weight_map.check_completeness(olm, hf, hf_ignored=ignored)

    def unique_numel(module) -> int:
        seen = set()
        total = 0
        for param in module.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    assert unique_numel(olm) == unique_numel(hf)


@pytest.mark.parity
@pytest.mark.parametrize("family", FAMILIES)
def test_tied_embeddings_share_storage(family):
    config = _load(family)
    olm, hf, _, _ = models.build_pair(family, config, device="cpu", init_seed=1)
    if family == "gpt2":
        emb = olm.blocks[0].blocks[0].embedding.weight
        head = olm.blocks[2].weight
    else:
        emb = olm.blocks[0].embedding.weight
        head = olm.blocks[3].weight
    assert emb is head
    # HF tie: lm_head.weight is same storage as embed_tokens / wte
    if family == "gpt2":
        assert hf.lm_head.weight.data_ptr() == hf.transformer.wte.weight.data_ptr()
    else:
        assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()


@pytest.mark.parity
@pytest.mark.parametrize("family", FAMILIES)
def test_forward_loss_gradient_parity(family):
    config = _load(family)
    models.set_determinism(11)
    olm, hf, weight_map, _ = models.build_pair(
        family, config, device="cpu", init_seed=11
    )
    tokens = compare.make_batch(
        11,
        config["batch"]["batch_size"],
        config["batch"]["seq_len"],
        config["model"]["vocab_size"],
        "cpu",
    )
    metrics = compare.compare_pair(
        olm, hf, weight_map, tokens, config["gradient_probes"]
    )
    # Observed clean-run errors are ~1e-7; keep a loose regression margin.
    assert metrics["max_logit_absolute_error"] < 1e-5
    assert metrics["mean_logit_absolute_error"] < 1e-6
    assert metrics["loss_absolute_error"] < 1e-5
    assert metrics["embedding_gradient_cosine"] > 0.999999
    assert metrics["early_layer_gradient_cosine"] > 0.999999
    assert metrics["late_layer_gradient_cosine"] > 0.999999

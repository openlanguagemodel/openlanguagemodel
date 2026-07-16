"""Paired OLM / Hugging Face model construction for parity experiments.

All models are built in FP32 on the requested device with dropout disabled.
The HF model is randomly initialized from a matched config (seeded), then its
weights are copied into the OLM model through the explicit weight map.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from benchmarks.demo2026.parity import weight_maps


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_hf_model(family: str, config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    hf_cfg = config.get("hf", {})

    if family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(
            vocab_size=model_cfg["vocab_size"],
            n_positions=model_cfg["max_seq_len"],
            n_embd=model_cfg["embed_dim"],
            n_layer=model_cfg["num_layers"],
            n_head=model_cfg["num_heads"],
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            activation_function=hf_cfg.get("activation_function", "gelu_new"),
            layer_norm_epsilon=hf_cfg.get("layer_norm_epsilon", 1e-5),
            attn_implementation="eager",
        )
        return GPT2LMHeadModel(cfg)

    if family == "llama3":
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=model_cfg["vocab_size"],
            hidden_size=model_cfg["embed_dim"],
            intermediate_size=model_cfg["intermediate_size"],
            num_hidden_layers=model_cfg["num_layers"],
            num_attention_heads=model_cfg["num_heads"],
            num_key_value_heads=model_cfg["num_kv_heads"],
            max_position_embeddings=model_cfg["max_seq_len"],
            rope_theta=model_cfg["rope_theta"],
            rope_scaling=None,  # standard RoPE only; see Llama3Model docstring
            rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
            attention_bias=hf_cfg.get("attention_bias", False),
            mlp_bias=hf_cfg.get("mlp_bias", False),
            attention_dropout=0.0,
            tie_word_embeddings=model_cfg.get("tie_weights", True),
            attn_implementation="eager",
        )
        return LlamaForCausalLM(cfg)

    if family == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        cfg = Qwen2Config(
            vocab_size=model_cfg["vocab_size"],
            hidden_size=model_cfg["embed_dim"],
            intermediate_size=model_cfg["intermediate_size"],
            num_hidden_layers=model_cfg["num_layers"],
            num_attention_heads=model_cfg["num_heads"],
            num_key_value_heads=model_cfg["num_kv_heads"],
            max_position_embeddings=model_cfg["max_seq_len"],
            rope_theta=model_cfg["rope_theta"],
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-6),
            attention_dropout=0.0,
            tie_word_embeddings=model_cfg.get("tie_weights", True),
            use_sliding_window=False,
            attn_implementation="eager",
        )
        return Qwen2ForCausalLM(cfg)

    raise ValueError(f"Unknown parity family: {family}")


def build_olm_model(family: str, config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]

    if family == "gpt2":
        from olm.models.openai.gpt2 import GPT2Model

        return GPT2Model(
            vocab_size=model_cfg["vocab_size"],
            embed_dim=model_cfg["embed_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            max_seq_len=model_cfg["max_seq_len"],
            dropout=model_cfg.get("dropout", 0.0),
        )

    if family == "llama3":
        from olm.models.meta.llama3 import Llama3Model

        return Llama3Model(
            vocab_size=model_cfg["vocab_size"],
            embed_dim=model_cfg["embed_dim"],
            intermediate_size=model_cfg["intermediate_size"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            num_kv_heads=model_cfg["num_kv_heads"],
            max_seq_len=model_cfg["max_seq_len"],
            rope_theta=model_cfg["rope_theta"],
            dropout=model_cfg.get("dropout", 0.0),
            tie_weights=model_cfg.get("tie_weights", True),
        )

    if family == "qwen2":
        from olm.models.alibaba.qwen2 import Qwen2Model

        return Qwen2Model(
            vocab_size=model_cfg["vocab_size"],
            embed_dim=model_cfg["embed_dim"],
            intermediate_size=model_cfg["intermediate_size"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            num_kv_heads=model_cfg["num_kv_heads"],
            max_seq_len=model_cfg["max_seq_len"],
            rope_theta=model_cfg["rope_theta"],
            dropout=model_cfg.get("dropout", 0.0),
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-6),
            tie_weights=model_cfg.get("tie_weights", True),
        )

    raise ValueError(f"Unknown parity family: {family}")


def build_pair(
    family: str, config: Dict[str, Any], device: str = "cpu", init_seed: int = 0
) -> Tuple[nn.Module, nn.Module, "weight_maps.WeightMap", list]:
    """Build (olm_model, hf_model, weight_map, hf_ignored) with copied weights."""
    torch.manual_seed(init_seed)
    hf_model = build_hf_model(family, config).to(device=device, dtype=torch.float32)
    hf_model.eval()

    olm_model = build_olm_model(family, config).to(device=device, dtype=torch.float32)
    olm_model.eval()

    map_module = {
        "gpt2": weight_maps.gpt2,
        "llama3": weight_maps.llama3,
        "qwen2": weight_maps.qwen2,
    }[family]
    weight_map = map_module.build_map(config["model"])
    hf_ignored = list(map_module.HF_IGNORED)

    weight_map.check_completeness(olm_model, hf_model, hf_ignored=hf_ignored)
    weight_map.copy_weights(hf_model, olm_model)
    return olm_model, hf_model, weight_map, hf_ignored

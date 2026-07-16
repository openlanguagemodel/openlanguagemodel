"""Forward / loss / gradient comparison between OLM and HF models.

Both models receive identical fixed token batches. The next-token
cross-entropy loss is computed *externally* from each model's logits with the
same shifted labels, so any loss difference is attributable purely to logits.
Gradients are compared as cosine similarities after transforming HF gradients
into OLM parameter layout via the same weight map used for copying.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F


def make_batch(
    seed: int, batch_size: int, seq_len: int, vocab_size: int, device: str
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=generator
    )
    return tokens.to(device)


def shifted_cross_entropy(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token CE: logits at position t predict token t+1."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na.item() == 0.0 and nb.item() == 0.0:
        return 1.0  # both zero: identical
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def compare_pair(
    olm_model,
    hf_model,
    weight_map,
    tokens: torch.Tensor,
    gradient_probes: Dict[str, str],
) -> Dict[str, Any]:
    """Run one fixed batch through both models and collect all parity metrics."""
    olm_model.zero_grad(set_to_none=True)
    hf_model.zero_grad(set_to_none=True)

    olm_logits = olm_model(tokens)
    hf_logits = hf_model(input_ids=tokens).logits

    diff = (olm_logits.detach() - hf_logits.detach()).abs()
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    olm_loss = shifted_cross_entropy(olm_logits, tokens)
    hf_loss = shifted_cross_entropy(hf_logits, tokens)
    loss_err = float((olm_loss.detach() - hf_loss.detach()).abs())

    olm_loss.backward()
    hf_loss.backward()

    olm_params = dict(olm_model.named_parameters())
    gradient_details: Dict[str, Any] = {}
    cosines: Dict[str, float] = {}
    for probe_name, olm_key in gradient_probes.items():
        olm_grad = olm_params[olm_key].grad
        if olm_grad is None:
            raise ValueError(f"OLM parameter {olm_key} has no gradient")
        hf_grad = weight_map.hf_gradient_for(olm_key, hf_model)
        cosine = _cosine(olm_grad, hf_grad)
        max_abs = float((olm_grad.double() - hf_grad.double()).abs().max())
        cosines[probe_name] = cosine
        gradient_details[probe_name] = {
            "olm_parameter": olm_key,
            "cosine": cosine,
            "max_absolute_error": max_abs,
            "olm_grad_norm": float(torch.linalg.norm(olm_grad.double())),
            "reference_grad_norm": float(torch.linalg.norm(hf_grad.double())),
        }

    record = {
        "max_logit_absolute_error": max_err,
        "mean_logit_absolute_error": mean_err,
        "loss_absolute_error": loss_err,
        "loss_olm": float(olm_loss.detach()),
        "loss_reference": float(hf_loss.detach()),
        "embedding_gradient_cosine": cosines["embedding"],
        "early_layer_gradient_cosine": cosines["early"],
        "late_layer_gradient_cosine": cosines["late"],
        "gradient_details": gradient_details,
    }
    for value in (max_err, mean_err, loss_err):
        if math.isnan(value) or math.isinf(value):
            record["status_hint"] = "error"
            break
    return record

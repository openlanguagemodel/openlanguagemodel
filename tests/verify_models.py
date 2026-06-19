"""Lightweight model-family verification.

This script is intentionally safe to run on laptops. It checks tiny
representatives for every implemented family and validates named preset
constructors without allocating the real multi-billion-parameter models.
"""

from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

import olm.models.alibaba.qwen2 as qwen2_module
import olm.models.allenai.olmo as olmo_module
import olm.models.facebook.opt as opt_module
import olm.models.google.gemma2 as gemma2_module
import olm.models.meta.llama2 as llama2_module
import olm.models.meta.llama3 as llama3_module
import olm.models.microsoft.phi3 as phi3_module
import olm.models.microsoft.phi4 as phi4_module
import olm.models.openai.gpt2 as gpt2_module
from olm.data.datasets import DataLoader
from olm.models.alibaba import Qwen2Model
from olm.models.allenai import OLMoModel
from olm.models.facebook import OPTModel
from olm.models.google import Gemma2Model
from olm.models.meta import Llama2Model, Llama3Model
from olm.models.microsoft import Phi3Model, Phi4Model
from olm.models.openai import GPT2Model
from olm.train import Trainer
from olm.train.optim import AdamW


def model_cases():
    return [
        (
            "GPT-2",
            GPT2Model(128, embed_dim=32, num_layers=1, num_heads=4, max_seq_len=16),
        ),
        (
            "Llama 2",
            Llama2Model(128, 32, 64, 1, 4, 4, 16),
        ),
        (
            "Llama 3",
            Llama3Model(128, 32, 64, 1, 4, 2, 16),
        ),
        (
            "Qwen 2.5",
            Qwen2Model(128, 32, 64, 1, 4, 2, 16, rope_theta=10000.0),
        ),
        (
            "Phi-3",
            Phi3Model(128, 32, 64, 1, 4, 4, 16),
        ),
        (
            "Phi-4",
            Phi4Model(128, 32, 64, 1, 4, 2, 16),
        ),
        (
            "Gemma 2",
            Gemma2Model(128, 32, 64, 1, 4, 2, 8, 16),
        ),
        (
            "OLMo",
            OLMoModel(128, 32, 64, 1, 4, 16),
        ),
        (
            "OPT",
            OPTModel(128, 32, 64, 1, 4, dropout=0.0),
        ),
    ]


def verify_forward_backward(name, model):
    model.train()
    input_ids = torch.randint(0, 128, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, 128), (name, logits.shape)
    logits.mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def verify_one_train_step(name, model):
    input_ids = torch.randint(0, 128, (4, 16))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=16,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )
    losses = trainer.train(epochs=1, max_steps=1, log_interval=100)
    assert len(losses) == 1, name
    assert torch.isfinite(torch.tensor(losses[0])), (name, losses)


def verify_named_presets_are_configured():
    cases = [
        (gpt2_module, "GPT2Model", ["GPT2", "GPT2Medium", "GPT2Large", "GPT2XL"]),
        (llama2_module, "Llama2Model", ["Llama2_7B", "Llama2_13B", "Llama2_70B"]),
        (
            llama3_module,
            "Llama3Model",
            [
                "Llama3_1_405B",
                "Llama3_1_70B",
                "Llama3_1_8B",
                "Llama3_2_3B",
                "Llama3_2_1B",
            ],
        ),
        (
            qwen2_module,
            "Qwen2Model",
            [
                "Qwen2_5_0_5B",
                "Qwen2_5_1_5B",
                "Qwen2_5_3B",
                "Qwen2_5_7B",
                "Qwen2_5_14B",
                "Qwen2_5_32B",
                "Qwen2_5_72B",
            ],
        ),
        (phi3_module, "Phi3Model", ["Phi3_5_Mini", "Phi3_Small"]),
        (phi4_module, "Phi4Model", ["Phi4_14B"]),
        (gemma2_module, "Gemma2Model", ["Gemma2_2B", "Gemma2_9B", "Gemma2_27B"]),
        (olmo_module, "OLMoModel", ["OLMo_7B"]),
        (opt_module, "OPTModel", ["OPT125M"]),
    ]

    for module, base_name, preset_names in cases:
        with patch.object(getattr(module, base_name), "__init__", return_value=None) as init:
            for preset_name in preset_names:
                getattr(module, preset_name)()

        assert init.call_count == len(preset_names), module.__name__
        for call in init.call_args_list:
            kwargs = call.kwargs
            assert kwargs["vocab_size"] > 0
            assert kwargs["embed_dim"] > 0
            assert kwargs["num_layers"] > 0
            assert kwargs["num_heads"] > 0


if __name__ == "__main__":
    print("Verifying tiny model-family trainability...")
    for name, model in model_cases():
        verify_forward_backward(name, model)
        print(f"  {name}: forward/backward ok")

    for name, model in model_cases():
        verify_one_train_step(name, model)
        print(f"  {name}: one train step ok")

    verify_named_presets_are_configured()
    print("  Named presets: constructor configs ok")
    print("Done.")

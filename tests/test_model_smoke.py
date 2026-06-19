import pytest
import torch
from unittest.mock import patch
from torch.utils.data import TensorDataset

from olm.data.datasets import DataLoader
import olm.models.alibaba.qwen2 as qwen2_module
import olm.models.allenai.olmo as olmo_module
import olm.models.facebook.opt as opt_module
import olm.models.google.gemma2 as gemma2_module
import olm.models.meta.llama2 as llama2_module
import olm.models.meta.llama3 as llama3_module
import olm.models.microsoft.phi3 as phi3_module
import olm.models.microsoft.phi4 as phi4_module
import olm.models.openai.gpt2 as gpt2_module
from olm.models.alibaba import Qwen2Model
from olm.models.allenai import OLMoModel, OLMo_7B
from olm.models.facebook import OPTModel
from olm.models.google import Gemma2Model
from olm.models.meta import Llama2Model, Llama3Model
from olm.models.microsoft import Phi3Model, Phi4Model, Phi4_14B
from olm.models.openai import GPT2Model
from olm.train import Trainer
from olm.train.optim import AdamW


def _model_cases():
    return [
        (
            "gpt2",
            GPT2Model(
                vocab_size=128,
                embed_dim=32,
                num_layers=1,
                num_heads=4,
                max_seq_len=16,
                dropout=0.0,
            ),
        ),
        (
            "llama2",
            Llama2Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=4,
                max_seq_len=16,
            ),
        ),
        (
            "llama3",
            Llama3Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=16,
            ),
        ),
        (
            "qwen2",
            Qwen2Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=16,
                rope_theta=10000.0,
                tie_weights=True,
            ),
        ),
        (
            "phi3_swiglu",
            Phi3Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=4,
                max_seq_len=16,
                activation="swiglu",
            ),
        ),
        (
            "phi3_geglu",
            Phi3Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=16,
                activation="geglu",
            ),
        ),
        (
            "phi4",
            Phi4Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=16,
            ),
        ),
        (
            "gemma2",
            Gemma2Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                max_seq_len=16,
            ),
        ),
        (
            "olmo",
            OLMoModel(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                max_seq_len=16,
            ),
        ),
        (
            "opt",
            OPTModel(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            ),
        ),
    ]


@pytest.mark.parametrize("name,model", _model_cases())
def test_model_family_forward_backward(name, model):
    del name
    model.train()
    input_ids = torch.randint(0, 128, (2, 16))

    logits = model(input_ids)
    assert logits.shape == (2, 16, 128)

    loss = logits.mean()
    loss.backward()

    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("name,model", _model_cases())
def test_model_family_trains_one_step(name, model):
    del name
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

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_phi_and_olmo_heads_are_not_tied_by_default():
    cases = [
        Phi3Model(128, 32, 64, 1, 4, 4, 16),
        Phi4Model(128, 32, 64, 1, 4, 2, 16),
        OLMoModel(128, 32, 64, 1, 4, 16),
    ]

    for model in cases:
        assert model.blocks[3].weight is not model.blocks[0].embedding.weight


def test_phi4_uses_reference_rope_theta():
    model = Phi4Model(128, 32, 64, 1, 4, 2, 16)
    block = model.blocks[1].stack[0]
    attn = block.blocks[0].block.blocks[1]
    assert attn.rope.base == 250000.0
    assert not attn.use_qk_norm

    with patch.object(phi4_module.Phi4Model, "__init__", return_value=None) as init:
        Phi4_14B()
    assert init.call_args.kwargs["rope_theta"] == 250000.0


def test_olmo_reference_vocab_size():
    with patch.object(olmo_module.OLMoModel, "__init__", return_value=None) as init:
        OLMo_7B()
    assert init.call_args.kwargs["vocab_size"] == 50280


def test_gemma2_embedding_is_scaled():
    model = Gemma2Model(128, 32, 64, 1, 4, 2, 8, 16)
    assert model.blocks[0].embed_scale == pytest.approx(32**0.5)


def test_gemma2_attention_reference_features():
    model = Gemma2Model(128, 32, 64, 2, 4, 2, 8, 16)
    first_block = model.blocks[1].stack[0]
    second_block = model.blocks[1].stack[1]

    assert first_block.sliding_window == 4096
    assert second_block.sliding_window is None
    assert not first_block.self_attn.use_qk_norm
    assert first_block.self_attn.attn_logit_softcap == 50.0
    assert first_block.self_attn.scale == pytest.approx(256**-0.5)
    assert model.blocks[4].softcap == 30.0

    logits = model(torch.randint(0, 128, (1, 8)))
    assert logits.abs().max().item() <= 30.0


def test_llama32_presets_tie_embeddings():
    model = Llama3Model(128, 32, 64, 1, 4, 2, 16, tie_weights=True)
    assert model.blocks[3].weight is model.blocks[0].embedding.weight

    with patch.object(llama3_module.Llama3Model, "__init__", return_value=None) as init:
        llama3_module.Llama3_2_1B()
        llama3_module.Llama3_2_3B()

    assert [call.kwargs["tie_weights"] for call in init.call_args_list] == [True, True]


def test_phi3_reference_preset_constants():
    with patch.object(phi3_module.Phi3Model, "__init__", return_value=None) as init:
        phi3_module.Phi3_5_Mini()
        phi3_module.Phi3_Small()

    mini, small = [call.kwargs for call in init.call_args_list]
    assert mini["max_seq_len"] == 131072
    assert small["intermediate_size"] == 14336
    assert small["max_seq_len"] == 131072
    assert small["rope_theta"] == 1000000.0


def test_qwen25_large_eps_matches_public_configs():
    with patch.object(qwen2_module.Qwen2Model, "__init__", return_value=None) as init:
        qwen2_module.Qwen2_5_14B()
        qwen2_module.Qwen2_5_32B()

    assert [call.kwargs["rms_norm_eps"] for call in init.call_args_list] == [
        1e-5,
        1e-5,
    ]


@pytest.mark.parametrize(
    "module,base_name,preset_names",
    [
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
    ],
)
def test_named_model_presets_call_base_constructor(module, base_name, preset_names):
    with patch.object(
        getattr(module, base_name), "__init__", return_value=None
    ) as init:
        for preset_name in preset_names:
            getattr(module, preset_name)()

    assert init.call_count == len(preset_names)
    for call in init.call_args_list:
        kwargs = call.kwargs
        assert kwargs["vocab_size"] > 0
        assert kwargs["embed_dim"] > 0
        assert kwargs["num_layers"] > 0
        assert kwargs["num_heads"] > 0
        if "max_seq_len" in kwargs:
            assert kwargs["max_seq_len"] > 0

import pytest
import torch
from unittest.mock import patch
from torch.utils.data import TensorDataset

from olm.data.datasets import DataLoader
import olm.models.alibaba.qwen2 as qwen2_module
import olm.models.alibaba.qwen3_next as qwen3_next_module
import olm.models.allenai.olmo as olmo_module
import olm.models.allenai.olmo3 as olmo3_module
import olm.models.facebook.opt as opt_module
import olm.models.google.gemma2 as gemma2_module
import olm.models.meta.llama2 as llama2_module
import olm.models.meta.llama3 as llama3_module
import olm.models.microsoft.phi3 as phi3_module
import olm.models.microsoft.phi4 as phi4_module
import olm.models.minimax.minimax_m2 as minimax_module
import olm.models.openai.gpt2 as gpt2_module
from olm.models.alibaba import Qwen2Model
from olm.models.allenai import OLMoModel, OLMo_7B, Olmo3Model
import olm.models.mistralai.mistral_large3 as mistral_large3_module
import olm.models.moonshotai.kimi_linear as kimi_linear_module
import olm.models.openai.gpt2 as gpt2_module
from olm.models.alibaba import Qwen2Model, Qwen3NextModel
from olm.models.allenai import OLMoModel, OLMo_7B
from olm.models.facebook import OPTModel
from olm.models.google import Gemma2Model
from olm.models.meta import Llama2Model, Llama3Model
from olm.models.microsoft import Phi3Model, Phi4Model, Phi4_14B
from olm.models.minimax import MiniMaxM2Model
from olm.models.mistralai import MistralLarge3Model
from olm.models.moonshotai import KimiLinearModel
from olm.models.openai import GPT2Model
from olm.nn.embeddings.positional.rope import PartialRotaryPositionalEmbedding
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
        (
            "olmo3",
            Olmo3Model(
                vocab_size=128,
                embed_dim=32,
                intermediate_size=64,
                num_layers=4,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                max_seq_len=16,
                sliding_window=4,
            ),
        ),
        (
            "mistral_large3",
            MistralLarge3Model(
                vocab_size=128,
                embed_dim=32,
                num_layers=4,
                num_heads=2,
                max_seq_len=16,
                kv_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                q_lora_rank=8,
                dense_intermediate_size=16,
                moe_intermediate_size=16,
                num_experts=4,
                num_shared_experts=1,
                top_k=2,
            ),
        ),
        (
            "qwen3_next",
            Qwen3NextModel(
                vocab_size=128,
                embed_dim=32,
                num_layers=4,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                max_seq_len=16,
                rms_norm_eps=1e-6,
                sliding_window=4,
            ),
        ),
        (
            "minimax_m2",
            MiniMaxM2Model(
                vocab_size=128,
                embed_dim=32,
                moe_intermediate_size=16,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=16,
                num_experts=4,
                top_k=2,
                head_dim=8,
                rope_theta=5000000.0,
                rotary_percentage=0.5,
                use_qk_norm=True,
            ),
        ),
        (
            "kimi_linear",
            KimiLinearModel(
                vocab_size=128,
                embed_dim=32,
                num_layers=4,
                kda_num_heads=4,
                kda_head_dim=8,
                kda_conv_kernel_size=4,
                mla_num_heads=4,
                max_seq_len=16,
                kv_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                q_lora_rank=8,
                dense_intermediate_size=16,
                intermediate_size=16,
                moe_intermediate_size=16,
                num_experts=4,
                num_shared_experts=1,
                top_k=2,
                first_k_dense_replace=3,
                full_attention_interval=4,
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


def _embedding_weight(model):
    if isinstance(model, GPT2Model):
        return model.blocks[0].blocks[0].embedding.weight
    return model.blocks[0].embedding.weight


def _head_weight(model):
    if isinstance(model, GPT2Model):
        return model.blocks[2].weight
    if isinstance(model, OPTModel):
        return model.blocks[5].weight
    return model.blocks[3].weight


@pytest.mark.parametrize("name,model", _model_cases())
def test_model_families_tie_heads_by_default(name, model):
    del name
    assert _head_weight(model) is _embedding_weight(model)


def test_model_families_can_disable_tied_heads():
    cases = [
        (
            Phi3Model(128, 32, 64, 1, 4, 4, 16, tie_weights=False),
            lambda model: model.blocks[3].weight,
        ),
        (
            Phi4Model(128, 32, 64, 1, 4, 2, 16, tie_weights=False),
            lambda model: model.blocks[3].weight,
        ),
        (
            OLMoModel(128, 32, 64, 1, 4, 16, tie_weights=False),
            lambda model: model.blocks[3].weight,
        ),
        (
            GPT2Model(128, 32, 1, 4, 16, tie_weights=False),
            lambda model: model.blocks[2].weight,
        ),
        (
            OPTModel(128, 32, 64, 1, 4, dropout=0.0, tie_weights=False),
            lambda model: model.blocks[5].weight,
        ),
    ]

    for model, head_weight in cases:
        assert head_weight(model) is not _embedding_weight(model)


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


def test_mistral_large3_uses_mla_with_dense_then_moe_layers():
    from olm.nn.attention import MultiHeadLatentAttention
    from olm.nn.feedforward import SwiGLUFFN, SwiGLUMoEFFN

    model = MistralLarge3Model(
        128, 32, 4, 2, 16, 8, 4, 4, 4, 8, 16, 16, 4, 1, 2, first_k_dense_replace=3
    )
    layers = model.blocks[1].blocks

    attn = layers[0].blocks[0].block.blocks[1]
    assert isinstance(attn, MultiHeadLatentAttention)

    ffns = [layer.blocks[1].block.blocks[1] for layer in layers]
    assert all(isinstance(f, SwiGLUFFN) for f in ffns[:3])
    assert isinstance(ffns[3], SwiGLUMoEFFN)
    assert ffns[3].num_shared_experts == 1


def test_mistral_large3_reference_preset_is_untied():
    with patch.object(
        mistral_large3_module.MistralLarge3Model, "__init__", return_value=None
    ) as init:
        mistral_large3_module.MistralLarge3()

    kwargs = init.call_args.kwargs
    assert kwargs["tie_weights"] is False
    assert kwargs["vocab_size"] == 131072
    assert kwargs["embed_dim"] == 7168
    assert kwargs["num_layers"] == 61
    assert kwargs["kv_lora_rank"] == 512
    assert kwargs["q_lora_rank"] == 1536
    assert kwargs["qk_rope_head_dim"] == 64
    assert kwargs["num_experts"] == 128
    assert kwargs["top_k"] == 4
    assert kwargs["first_k_dense_replace"] == 3
def test_qwen3_next_alternates_linear_and_full_attention():
    from olm.nn.attention import GatedAttention, GatedDeltaNet

    model = Qwen3NextModel(
        128, 32, 8, 4, 2, 8, 16, 4, 4, 8, 8, 4, 16, 4, 1, 2, full_attention_interval=4
    )
    layers = model.blocks[1].blocks
    attn_types = [type(layer.blocks[0].block.blocks[1]) for layer in layers]

    # 3 linear (Gated DeltaNet) layers, then 1 full (Gated Attention) layer, repeated.
    assert attn_types == [
        GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, GatedAttention,
        GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, GatedAttention,
    ]


def test_qwen3_next_reference_preset_is_untied():
    with patch.object(
        qwen3_next_module.Qwen3NextModel, "__init__", return_value=None
    ) as init:
        qwen3_next_module.Qwen3Next80BA3B()

    kwargs = init.call_args.kwargs
    assert kwargs["tie_weights"] is False
    assert kwargs["vocab_size"] == 151936
    assert kwargs["embed_dim"] == 2048
    assert kwargs["num_layers"] == 48
    assert kwargs["num_experts"] == 512
    assert kwargs["top_k"] == 10
    assert kwargs["partial_rotary_factor"] == 0.25
    assert kwargs["full_attention_interval"] == 4


def test_kimi_linear_uses_kda_with_periodic_mla_and_dense_then_moe_layers():
    from olm.nn.attention import KimiDeltaAttention, MultiHeadLatentAttention
    from olm.nn.feedforward import SwiGLUFFN, SwiGLUMoEFFN

    model = KimiLinearModel(
        128, 32, 6, 4, 8, 4, 4, 16, 8, 4, 4, 4, None, 16, 16, 4, 1, 2,
        full_attention_interval=4, first_k_dense_replace=1,
    )
    layers = model.blocks[1].blocks
    attn_types = [type(layer.blocks[0].block.blocks[1]) for layer in layers]

    # Layer 4 (1-indexed) hits the modulo; the final layer (6) is always MLA too.
    assert attn_types == [
        KimiDeltaAttention, KimiDeltaAttention, KimiDeltaAttention,
        MultiHeadLatentAttention, KimiDeltaAttention, MultiHeadLatentAttention,
    ]

    ffns = [layer.blocks[1].block.blocks[1] for layer in layers]
    assert isinstance(ffns[0], SwiGLUFFN)
    assert all(isinstance(f, SwiGLUMoEFFN) for f in ffns[1:])
    assert all(isinstance(f.router, kimi_linear_module.KimiLinearRouter) for f in ffns[1:])


def test_kimi_linear_48b_reference_preset():
    with patch.object(
        kimi_linear_module.KimiLinearModel, "__init__", return_value=None
    ) as init:
        kimi_linear_module.KimiLinear48BA3B()

    kwargs = init.call_args.kwargs
    assert kwargs["tie_weights"] is False
    assert kwargs["vocab_size"] == 163840
    assert kwargs["embed_dim"] == 2304
    assert kwargs["num_layers"] == 27
    assert kwargs["q_lora_rank"] is None
    assert kwargs["kv_lora_rank"] == 512
    assert kwargs["qk_rope_head_dim"] == 64
    assert kwargs["num_experts"] == 256
    assert kwargs["top_k"] == 8
    assert kwargs["routed_scaling_factor"] == 2.446
    assert kwargs["first_k_dense_replace"] == 1


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


def test_llama3_ties_embeddings_by_default():
    model = Llama3Model(128, 32, 64, 1, 4, 2, 16)
    assert model.blocks[3].weight is model.blocks[0].embedding.weight

    with patch.object(llama3_module.Llama3Model, "__init__", return_value=None) as init:
        llama3_module.Llama3_2_1B()
        llama3_module.Llama3_2_3B()

    assert [call.kwargs.get("tie_weights", True) for call in init.call_args_list] == [
        True,
        True,
    ]


def test_llama3_2_1b_reference_config():
    with patch.object(llama3_module.Llama3Model, "__init__", return_value=None) as init:
        llama3_module.Llama3_2_1B()

    kwargs = init.call_args.kwargs
    assert kwargs["embed_dim"] == 2048
    assert kwargs["intermediate_size"] == 8192
    assert kwargs["num_layers"] == 16
    assert kwargs["num_heads"] == 32
    assert kwargs["num_kv_heads"] == 8
    assert kwargs["max_seq_len"] == 131072
    assert kwargs["rope_theta"] == 500000.0


def test_olmo3_uses_post_norm_and_sliding_window_pattern():
    model = Olmo3Model(128, 32, 64, 4, 4, 2, 8, max_seq_len=16, sliding_window=4)
    blocks = model.blocks[1].stack

    # 3 sliding-window layers followed by 1 full-attention (global) layer.
    assert [b.sliding_window for b in blocks] == [4, 4, 4, None]
    # Post-norm: per-sublayer output norms, QK-norm enabled, no input norm.
    first = blocks[0]
    assert hasattr(first, "post_attention_layernorm")
    assert hasattr(first, "post_feedforward_layernorm")
    assert not hasattr(first, "input_layernorm")
    assert first.self_attn.use_qk_norm


def test_olmo3_reference_presets_are_untied():
    with patch.object(
        olmo3_module.Olmo3Model, "__init__", return_value=None
    ) as init:
        olmo3_module.Olmo3_7B()
        olmo3_module.Olmo3_32B()

    seven_b, thirty_two_b = [call.kwargs for call in init.call_args_list]
    assert seven_b["tie_weights"] is False
    assert seven_b["num_heads"] == 32 and seven_b["num_kv_heads"] == 32
    assert seven_b["vocab_size"] == 100278
    assert thirty_two_b["tie_weights"] is False
    assert thirty_two_b["num_heads"] == 40 and thirty_two_b["num_kv_heads"] == 8
    assert thirty_two_b["intermediate_size"] == 27648
    assert thirty_two_b["rope_theta"] == 500000.0


def test_minimax_m2_uses_partial_rope_and_sigmoid_router():
    model = MiniMaxM2Model(128, 32, 16, 1, 4, 2, 16, 4, 2, head_dim=8)
    block = model.blocks[1].stack[0]
    attn = block.blocks[0].block.blocks[1]
    moe = block.blocks[1].block.blocks[1]

    assert isinstance(attn.rope, PartialRotaryPositionalEmbedding)
    # head_dim=8, rotary_percentage=0.5 -> 4 rotated dims
    assert attn.rope.rotary_dim == 4
    assert attn.use_qk_norm
    assert isinstance(moe.router, minimax_module.MiniMaxM2Router)


def test_minimax_m2_reference_preset_is_untied():
    with patch.object(
        minimax_module.MiniMaxM2Model, "__init__", return_value=None
    ) as init:
        minimax_module.MiniMaxM2()

    kwargs = init.call_args.kwargs
    assert kwargs["tie_weights"] is False
    assert kwargs["vocab_size"] == 200064
    assert kwargs["head_dim"] == 128
    assert kwargs["num_experts"] == 256
    assert kwargs["top_k"] == 8
    assert kwargs["rope_theta"] == 5000000.0
    assert kwargs["rotary_percentage"] == 0.5


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


def test_qwen25_1_5b_reference_config():
    with patch.object(qwen2_module.Qwen2Model, "__init__", return_value=None) as init:
        qwen2_module.Qwen2_5_1_5B()

    kwargs = init.call_args.kwargs
    assert kwargs["embed_dim"] == 1536
    assert kwargs["intermediate_size"] == 8960
    assert kwargs["num_layers"] == 28
    assert kwargs["num_heads"] == 12
    assert kwargs["num_kv_heads"] == 2
    assert kwargs["max_seq_len"] == 131072
    assert kwargs["rope_theta"] == 1000000.0


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
        (olmo3_module, "Olmo3Model", ["Olmo3_7B", "Olmo3_32B"]),
        (opt_module, "OPTModel", ["OPT125M"]),
        (minimax_module, "MiniMaxM2Model", ["MiniMaxM2"]),
        (mistral_large3_module, "MistralLarge3Model", ["MistralLarge3"]),
        (qwen3_next_module, "Qwen3NextModel", ["Qwen3Next80BA3B"]),
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

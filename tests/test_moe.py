import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn
from olm.nn.feedforward import ClassicMoEFFN, GeGLUMoEFFN, SwiGLUMoEFFN
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.moe import MoEFeedForward, MoERouter
from olm.train.losses import SequenceLoadBalanceLoss

def test_moe_ffn():
    print("Testing MoE FFNs...")
    batch_size = 2
    seq_len = 10
    embed_dim = 32
    num_experts = 4
    top_k = 2
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # List of classes to test
    classes = [ClassicMoEFFN, GeGLUMoEFFN, SwiGLUMoEFFN]
    
    for cls in classes:
        print(f"Testing {cls.__name__}...")
        
        # 1. Basic Instantiation with Fine-Grained (Shared Experts)
        model = cls(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_shared_experts=2, # Fine-grained / Shared experts test
            top_k=top_k,
            hidden_dim=embed_dim * 2, # Smaller hidden dim for test
            dropout=0.1
        )
        
        # 2. Forward Pass
        out = model(x)
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
        
        # 3. Backward Pass (Gradient Flow)
        loss = out.sum()
        loss.backward()
        
        # Check if router gate has grads
        assert model.router.gate.weight.grad is not None, "Router gate weight has no gradient!"
        
        # Check if experts have grads
        # Note: Since input is random, all experts might not be selected, but with top_k=2 and 4 experts, 
        # and batch*seq=20, it's highly likely all are selected.
        # But ensure at least some experts have grads.
        expert_grads = False
        for i, expert in enumerate(model.experts):
             # Depending on implementation, parameters might be nested differently
             # Just check one parameter
             for p in expert.parameters():
                 if p.grad is not None:
                     expert_grads = True
                     break
        assert expert_grads, "No gradients flow to experts!"

        # Check shared experts grads
        shared_grads = False
        for expert in model.shared_experts:
            for p in expert.parameters():
                if p.grad is not None:
                    shared_grads = True
                    break
        assert shared_grads, "No gradients flow to shared experts!"
        
        print(f"{cls.__name__} Passed!")

if __name__ == "__main__":
    try:
        test_moe_ffn()
        print("\nAll MoE Tests Passed!")
    except Exception as e:
        print(f"\nTests Failed: {e}")
        import traceback
        traceback.print_exc()


def test_canonical_moe_records_router_stats_and_backpropagates():
    torch.manual_seed(0)
    model = MoEFeedForward(
        embed_dim=16,
        expert_cls=SwiGLUFFN,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        expert_kwargs={"hidden_dim": 32, "dropout": 0.0},
        scoring_func="sigmoid",
        routing_method="noaux_tc",
        use_router_bias=True,
        fp32_gate=True,
    )
    x = torch.randn(2, 5, 16)

    out, router_logits = model(x)
    stats = model.get_router_stats()

    assert out.shape == x.shape
    assert router_logits.shape == (2, 5, 4)
    assert stats is not None
    assert stats.top_k_indices.shape == (2, 5, 2)
    assert stats.top_k_weights.shape == (2, 5, 2)
    assert stats.expert_fraction.shape == (4,)

    out.mean().backward()
    assert model.router.gate.weight.grad is not None


def test_moe_router_aux_loss_free_bias_update_moves_underused_experts_up():
    router = MoERouter(
        embed_dim=8,
        num_experts=4,
        top_k=1,
        routing_method="noaux_tc",
    )
    before = router.expert_bias.detach().clone()
    expert_fraction = torch.tensor([0.7, 0.1, 0.1, 0.1])

    router.update_expert_bias_(expert_fraction, update_rate=0.1)

    assert router.expert_bias[0] < before[0]
    assert torch.all(router.expert_bias[1:] > before[1:])


def test_sequence_load_balance_loss_supports_sigmoid_router_stats():
    router_logits = torch.randn(2, 4, 3, requires_grad=True)
    top_k_indices = torch.tensor(
        [
            [[0, 1], [1, 2], [0, 2], [1, 2]],
            [[2, 0], [0, 1], [1, 2], [0, 2]],
        ]
    )
    loss_fn = SequenceLoadBalanceLoss(
        num_experts=3,
        top_k=2,
        scoring_func="sigmoid",
        coefficient=0.01,
    )

    loss = loss_fn(router_logits, top_k_indices)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert router_logits.grad is not None


def test_moe_ffn_shared_experts_can_be_wider_than_routed_experts():
    model = ClassicMoEFFN(
        embed_dim=32,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        hidden_dim=16,
        shared_hidden_dim=64,
        bias=False,
    )

    assert model.experts[0].hidden_dim == 16
    assert model.shared_experts[0].hidden_dim == 64

    x = torch.randn(2, 5, 32)
    assert model(x).shape == x.shape


def test_moe_router_applies_routed_scaling_factor():
    model = ClassicMoEFFN(
        embed_dim=32, num_experts=4, top_k=2, hidden_dim=16, bias=False
    )
    x = torch.randn(2, 5, 32)

    model.router.routed_scaling_factor = 1.0
    baseline = model.compute_routed(x)
    model.router.routed_scaling_factor = 2.5
    scaled = model.compute_routed(x)

    assert torch.allclose(scaled, 2.5 * baseline, atol=1e-5)


def test_latent_moe_routes_in_latent_space_with_full_width_shared_experts():
    from olm.nn.feedforward import LatentMoEFFN
    from olm.nn.feedforward.moe_base import MoEFeedForwardBase

    model = LatentMoEFFN(
        embed_dim=32,
        latent_dim=8,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        hidden_dim=16,
        shared_hidden_dim=64,
        bias=False,
        routed_scaling_factor=5.0,
    )

    assert isinstance(model, MoEFeedForwardBase)
    # Routed experts live in the bottleneck...
    assert model.experts[0].embed_dim == 8
    # ...while the router and the shared experts stay at full model width.
    assert model.router.gate.in_features == 32
    assert model.router_embed_dim == 32
    assert model.shared_experts[0].embed_dim == 32
    assert model.shared_experts[0].hidden_dim == 64
    assert model.embed_dim == 32

    x = torch.randn(2, 5, 32)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert model.down_proj.weight.grad is not None
    assert model.up_proj.weight.grad is not None


def test_mamba2_mixer_fills_the_attention_base_role():
    import pytest
    from olm.nn.attention import Mamba2Mixer
    from olm.nn.attention.base import AttentionBase

    mixer = Mamba2Mixer(32, num_heads=4, head_dim=8, state_size=16, n_groups=1)
    assert isinstance(mixer, AttentionBase)

    x = torch.randn(2, 5, 32)
    assert mixer(x).shape == x.shape

    with pytest.raises(NotImplementedError):
        mixer.compute_attention(x, x, x)


def test_latent_moe_scores_routing_on_the_full_width_hidden_state():
    """Routing must see the uncompressed hidden state, not down_proj(x)."""
    from olm.nn.feedforward import LatentMoEFFN

    model = LatentMoEFFN(
        embed_dim=32, latent_dim=8, num_experts=4, top_k=2, hidden_dim=16, bias=False
    )
    x = torch.randn(2, 5, 32)

    model(x)
    logits_from_forward = model.last_router_logits.clone()

    # The gate only accepts full-width input, and the logits recorded during
    # forward are exactly the ones it produces from x.
    assert model.router.gate.in_features == 32
    assert torch.allclose(logits_from_forward, model.router(x)[2], atol=1e-6)


def test_moe_router_group_limited_routing_restricts_experts_to_top_groups():
    router = MoERouter(
        embed_dim=16,
        num_experts=8,
        top_k=2,
        scoring_func="sigmoid",
        n_group=4,
        topk_group=1,
    )
    x = torch.randn(2, 5, 16)

    top_k_indices, top_k_weights, _ = router(x)

    # With one group of two experts eligible, every token must draw both of
    # its experts from the same group.
    groups = top_k_indices // 2
    assert (groups[..., 0] == groups[..., 1]).all()
    assert torch.isfinite(top_k_weights).all()
    assert router.last_stats.metadata["n_group"] == 4


def test_moe_router_group_limited_routing_validates_config():
    import pytest

    with pytest.raises(ValueError):
        MoERouter(embed_dim=16, num_experts=8, n_group=4)  # missing topk_group
    with pytest.raises(ValueError):
        MoERouter(embed_dim=16, num_experts=7, n_group=4, topk_group=1)
    with pytest.raises(ValueError):
        MoERouter(embed_dim=16, num_experts=8, n_group=4, topk_group=5)


def test_nemotron_moe_layers_use_sigmoid_correction_bias_routing():
    from olm.models.nvidia import NemotronHModel
    from olm.nn.feedforward import ClassicMoEFFN as _ClassicMoEFFN

    model = NemotronHModel(
        vocab_size=64,
        embed_dim=32,
        hybrid_override_pattern="ME",
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=1,
        conv_kernel_size=4,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=32,
        routed_scaling_factor=2.5,
        n_group=4,
        topk_group=2,
        tie_weights=False,
    )

    moe_layers = [m for m in model.modules() if isinstance(m, _ClassicMoEFFN)]
    assert moe_layers
    for moe in moe_layers:
        router = moe.router
        assert router.scoring_func == "sigmoid"
        assert router.routing_method == "noaux_tc"
        # Auxiliary-loss-free correction bias, applied to selection only.
        assert router.expert_bias is not None
        assert router.n_group == 4 and router.topk_group == 2
        assert router.routed_scaling_factor == 2.5

    logits = model(torch.randint(0, 64, (2, 6)))
    assert logits.shape == (2, 6, 64)


def test_mamba2_dt_bias_matches_reference_timestep_initialization():
    import math
    import torch.nn.functional as F
    from olm.nn.attention import Mamba2Mixer

    time_step_min, time_step_max = 0.001, 0.1
    mixer = Mamba2Mixer(
        32,
        num_heads=16,
        head_dim=8,
        state_size=16,
        n_groups=1,
        time_step_min=time_step_min,
        time_step_max=time_step_max,
    )

    # dt_bias stores inverse-softplus timesteps, so softplus recovers the
    # sampled timesteps -- which must lie inside the configured range.
    dt = F.softplus(mixer.dt_bias)
    assert (dt >= time_step_min - 1e-6).all()
    assert (dt <= time_step_max + 1e-6).all()
    # A zero bias would put every head at softplus(0) ~ 0.693 instead.
    assert not torch.allclose(mixer.dt_bias, torch.zeros_like(mixer.dt_bias))
    assert dt.std() > 0

    narrow = Mamba2Mixer(
        32,
        num_heads=8,
        head_dim=8,
        state_size=16,
        n_groups=1,
        time_step_min=0.05,
        time_step_max=0.05,
    )
    assert torch.allclose(
        F.softplus(narrow.dt_bias), torch.full((8,), 0.05), atol=1e-5
    )
    assert math.isclose(narrow.time_step_max, 0.05)

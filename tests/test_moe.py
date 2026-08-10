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
    # Router and routed experts live in the bottleneck...
    assert model.router.gate.in_features == 8
    assert model.experts[0].embed_dim == 8
    # ...while the shared experts stay at full model width.
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

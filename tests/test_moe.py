import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn
from olm.nn.feedforward import ClassicMoEFFN, GeGLUMoEFFN, SwiGLUMoEFFN

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

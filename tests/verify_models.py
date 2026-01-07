
import torch
import torch.nn as nn
from olm.models.meta import Llama3_1_8B, Llama2_7B
from olm.models.alibaba import Qwen2_5_7B
from olm.models.google import Gemma2_9B
from olm.models.microsoft import Phi3_5_Mini
from olm.models.allenai import OLMo_7B

def test_model(model_cls, name):
    print(f"Testing {name}...")
    try:
        model = model_cls()
        model.eval()
        
        # Reduced seq len for speed
        x = torch.randint(0, 1000, (1, 64)) 
        
        with torch.no_grad():
            out = model(x)
            
        print(f"  Shape: {out.shape}")
        # Expected: (1, 64, vocab_size)
        
        assert out.shape[0] == 1
        assert out.shape[1] == 64
        # Check vocab size?
        # print(f"  Vocab: {out.shape[2]}")
        print(f"  {name} Passed ✓")
        
    except Exception as e:
        print(f"  {name} Failed ✗")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Verifying Models Implementation...")
    
    test_model(Llama3_1_8B, "Llama 3.1 8B")
    test_model(Llama2_7B, "Llama 2 7B")
    test_model(Qwen2_5_7B, "Qwen 2.5 7B")
    test_model(Gemma2_9B, "Gemma 2 9B")
    test_model(Phi3_5_Mini, "Phi 3.5 Mini")
    test_model(OLMo_7B, "OLMo 7B")
    
    print("Done.")

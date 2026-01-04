
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import re
import torch
import torch.nn as nn

# Import from new modules
from olm.models.google.gemma2 import Gemma2Block
from olm.models.meta.llama3 import Llama3Model, Llama3_2_1B
from olm.models.alibaba.qwen2 import Qwen2Model, Qwen2_5_3B
from olm.models.allenai.olmo import OLMoModel, OLMo_7B
from olm.models.microsoft.phi4 import Phi4Model, Phi4_14B
from olm.models.openai.gpt2 import GPT2

# Check if old modules are gone (optional, but good)
def check_absence():
    paths = [
        'src/olm/models/meta/llama.py',
        'src/olm/models/alibaba/qwen.py',
        'src/olm/models/microsoft/phi.py',
        'src/olm/models/google/gemma.py',
        'src/olm/models/openai/gpt.py'
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"WARNING: Old file {p} still exists!")
        else:
            print(f"Verified deletion: {p}")

def verify_gemma_sandwich():
    print("\nVerifying Gemma2Block Sandwich Norm...")
    block = Gemma2Block(
        embed_dim=256,
        intermediate_size=1024,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=128,
        dropout=0.0,
        rope_theta=10000.0
    )
    # Check composition
    # Block 0: Attn Sublayer -> [Residual(Block([PreNorm, GQA])), PostNorm]
    # Block 1: MLP Sublayer -> [Residual(Block([PreNorm, GeGLU])), PostNorm]
    
    print(f"  Block Structure: {block}")
    
    # We implicitly trust the Block logic if it runs, but let's check for "sandwich" visual
    # It prints the structure.

def verify_instantiations():
    print("\nVerifying Model Instantiations...")
    try:
        # l3 = Llama3_2_1B()
        l3 = Llama3Model(vocab_size=1000, embed_dim=256, intermediate_size=1024, num_layers=2, num_heads=4, num_kv_heads=4, max_seq_len=128, rope_theta=10000.0)
        print("  Llama 3.2 (Mini) instantiated.")
        
        # q2 = Qwen2_5_3B()
        q2 = Qwen2Model(vocab_size=1000, embed_dim=256, intermediate_size=1024, num_layers=2, num_heads=4, num_kv_heads=4, max_seq_len=128, rope_theta=10000.0)
        print("  Qwen 2.5 (Mini) instantiated.")
        # p4 = Phi4_14B() # Too big for local test
        p4 = Phi4Model(vocab_size=1000, embed_dim=256, intermediate_size=1024, num_layers=2, num_heads=4, num_kv_heads=4, max_seq_len=128, rope_theta=10000.0)
        print("  Phi-4 (Mini) instantiated.")
        
        gp = GPT2()
        print("  GPT-2 instantiated.")
        
        # ol = OLMo_7B() # Too big for local test
        ol = OLMoModel(vocab_size=1000, embed_dim=256, intermediate_size=1024, num_layers=2, num_heads=4, max_seq_len=128, dropout=0.0)
        print("  OLMo (Mini) instantiated.")
    except Exception as e:
        print(f"  FAILED instantiation: {e}")
        raise e

if __name__ == "__main__":
    check_absence()
    verify_gemma_sandwich()
    verify_instantiations()

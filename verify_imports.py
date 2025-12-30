import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.abspath("src"))

try:
    print("Importing olm...")
    import olm
    print("Importing olm.nn...")
    import olm.nn
    print("Importing olm.nn.blocks...")
    from olm.nn.blocks import TransformerBlock, LM, OutputHead
    print("Importing olm.nn.attention...")
    from olm.nn.attention import MultiHeadAttention, MultiHeadAttentionwithRoPE
    print("Importing olm.nn.feedforward...")
    from olm.nn.feedforward import SwiGLUFFN, ClassicFFN
    print("Importing olm.nn.structure...")
    from olm.nn.structure import Block
    print("Importing olm.nn.structure.combinators...")
    from olm.nn.structure.combinators import Repeat, Residual, Parallel
    print("All imports successful!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

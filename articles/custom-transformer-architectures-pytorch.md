# Custom Transformer Architectures in PyTorch Without Rewriting the Training Loop

Architecture research is often slowed down by code ownership problems. The
thing you want to test is small: a new attention rule, a different norm, a
feed-forward variant, or a changed residual pattern. The file you have to edit
is huge.

OLM is built around the opposite idea. Components are ordinary PyTorch modules.
Wiring is expressed with `Block`, `Residual`, `Repeat`, and `Parallel`.

## Swap the Attention Rule

```python
import torch
from olm.nn.attention import AttentionBase

class LocalWindowAttention(AttentionBase):
    def __init__(self, embed_dim, num_heads, window=256):
        super().__init__(embed_dim, num_heads)
        self.window = window

    def compute_attention(self, q, k, v, mask=None):
        scores = (q @ k.transpose(-2, -1)) * self.scale
        seq = q.size(-2)
        pos = torch.arange(seq, device=q.device)
        local = (pos[:, None] - pos[None, :]).abs() <= self.window
        causal = pos[:, None] >= pos[None, :]
        scores = scores.masked_fill(~(local & causal), float("-inf"))
        probs = self.dropout(scores.softmax(dim=-1))
        return probs @ v
```

This is not a production recommendation. It is the point of the interface: the
experiment is local. You can put this attention module inside an OLM block, run
it through the same trainer, and compare against the baseline.

## Keep the Training Path Stable

Once the model is a `torch.nn.Module`, it can run through OLM's `Trainer`,
`DDPTrainer`, `FSDPTrainer`, or your own loop. That means your ablation does not
need to own data loading, checkpointing, logging, gradient accumulation, AMP, or
distributed setup.

For the full architecture guide, read
[The Block System](/docs/guides/architecture/). For component choices, read
[Building Blocks](/docs/guides/components/).

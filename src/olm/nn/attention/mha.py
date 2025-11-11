import torch, torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from base import AttentionBase

class MultiHeadAttention(AttentionBase):
    def __init__(self, embed_dims, num_heads, dropout=0.0, causal=False):
        super().__init__(embed_dims, num_heads, dropout)
        self.causal = causal

    def compute_attention(self, q, k, v, mask=None):
        # q, k, v: [batch, heads, seq, dim]
        # doing scaled dot product attention here

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        out = torch.matmul(attention_probs, v)
        return out
# src/olm/nn/activations/swiglu.py
import torch, torch.nn as nn
from olm.core.registry import ACTIVATIONS


@ACTIVATIONS.register("swiglu")
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        swish_gate = gate * torch.sigmoid(gate)
        return x * swish_gate

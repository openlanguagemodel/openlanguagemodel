# src/olm/nn/activations/gelu.py
import torch, torch.nn as nn
from olm.core.registry import ACTIVATIONS


@ACTIVATIONS.register("gelu")
class GeLU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

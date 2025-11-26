import torch
import torch.nn as nn


class QKVProjection(nn.Module):

    def __init__(self, dim_in, dim_q, dim_k, dim_v, bias=True, init="xavier"):
        super().__init__()

        self.W_q = nn.Linear(dim_in, dim_q, bias=bias)
        self.W_k = nn.Linear(dim_in, dim_k, bias=bias)
        self.W_v = nn.Linear(dim_in, dim_v, bias=bias)

        layers = [self.W_q, self.W_k, self.W_v]

        # optional initialization
        for layer in layers:
            if init == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init == "kaiming":
                nn.init.kaiming_uniform_(layer.weight)
            elif init == "normal":
                nn.init.normal_(layer.weight, std=0.02)
            else:
                raise ValueError(f"Unknown init: {init}")

            if bias:
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return Q, K, V
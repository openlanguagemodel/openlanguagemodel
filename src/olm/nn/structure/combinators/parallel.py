from base import BaseCombinator
import torch

class Parallel(BaseCombinator):
    def __init__(self, blocks, merge='ADD', dim=-1):
        super().__init__()

        self.blocks = blocks
        self.merge = merge
        self.dim = dim

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block.forward(x))

        # implementing ADD
        if self.merge == 'ADD':
            y = torch.sum(outputs, dim=self.dim)
        
        # implementing CONCAT
        elif self.merge == 'CONCAT':
            y = torch.cat(outputs, dim=self.dim)

        # implementing MATMUL
        elif self.merge == 'MATMUL':
            assert len(outputs) == 2
            assert outputs[0].shape[-1] == outputs[1].shape[-2]
            y = torch.einsum('...xy,...yz->...xz', outputs[0], outputs[1])

        return y



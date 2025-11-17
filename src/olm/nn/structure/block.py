class Block:
    def __init__(self, blocks):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        
        return x
from base import BaseCombinator

class Residual(BaseCombinator):
    def __init__(self, block):
        super().__init__()

        self.block = block

    def forward(self, x):
        y = x+self.block(x)
        return y
    
    
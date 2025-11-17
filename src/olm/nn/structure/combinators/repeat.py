from base import BaseCombinator


# note that module_func has to be a lambda function
class Repeat(BaseCombinator):
    def __init__(self, module_func, num_repeat):
        super().__init__()

        self.module = module_func
        self.num_repeat = num_repeat

        self.stack = [module_func() for _ in range(num_repeat)]

    def forward(self, x):
        for block in self.stack:
            x = block.forward(x)
        return x
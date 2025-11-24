import torch


class MergeFunction:
    def forward(self):
        pass


class Add(MergeFunction):
    def forward(self, modules, dim):
        y = torch.sum(modules, dim=dim)
        return y
    

class Concat(MergeFunction):
    def forward(self, modules, dim):
        y = torch.concat(modules, dim=dim)
        return y


class Matmul(MergeFunction):
    def forward(self, modules, dim):
        y = torch.einsum('...')


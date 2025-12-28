from typing import Type
from olm.data.tokenization import TokenizerBase
from olm.nn.structure.pipeline import Pipeline
from olm.data.datasets import Dataset
import torch.optim
import torch
from torch.cuda.amp import autocast, GradScaler
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.base import LossBase

class Trainer:
    def __init__(
        self,
        model: Type[Pipeline],
        optimizer: Type[torch.optim.Optimizer],
        dataloader: Type[Dataset],
        device: str,
        context_length: int,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        loss: Type[LossBase] = CrossEntropyLoss,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.context_length = context_length
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.loss = loss()

    def train(self, epochs: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(epochs):
            for step, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(x)  # (B, T, V)
                    loss = self.loss(logits, y)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

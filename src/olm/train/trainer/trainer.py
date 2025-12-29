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
    """
    Manages the training loop for Open Language Model (OLM) architectures.

    This trainer handles the core training logic including:
    - Automatic Mixed Precision (AMP) scaling
    - Gradient accumulation
    - Device management (moving data/models to GPU)
    - Optimization steps

    Attributes:
        model (Pipeline): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (Dataset): The data provider.
        device (str): The device to train on (e.g., 'cuda', 'cpu').
        context_length (int): The maximum sequence length for training.
        grad_accum_steps (int): Number of steps to accumulate gradients before updating.
        use_amp (bool): Whether to use Automatic Mixed Precision.
        scaler (GradScaler): Gradient scaler for AMP.
        loss (LossBase): The loss function instance.
    """
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
        """
        Initializes the Trainer.

        Args:
            model (Type[Pipeline]): The model architecture to train.
            optimizer (Type[torch.optim.Optimizer]): The optimizer class or instance.
            dataloader (Type[Dataset]): The dataset iterator.
            device (str): Target device ('cuda' or 'cpu').
            context_length (int): Maximum sequence length.
            grad_accum_steps (int, optional): Steps for gradient accumulation. Defaults to 1.
            use_amp (bool, optional): Enable Automatic Mixed Precision. Defaults to True.
            loss (Type[LossBase], optional): Loss function class. Defaults to CrossEntropyLoss.
        """
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
        """
        Executes the training loop for a specified number of epochs.

        Iterates through the dataloader, computes loss, scales gradients (if AMP is enabled),
        and updates model parameters. Handles gradient accumulation.

        Args:
            epochs (int): The number of complete passes through the dataset.

        Side Effects:
            - Updates `self.model` parameters.
            - Prints training progress (implicit in loop, though not currently implemented).
            - Modifies optimizer state.
        """
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

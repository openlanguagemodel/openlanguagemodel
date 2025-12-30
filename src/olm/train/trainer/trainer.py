from typing import Type
from olm.data.tokenization import TokenizerBase
from olm.nn.structure.pipeline import Pipeline
from olm.data.datasets import Dataset
import torch.optim
import torch
from torch.amp import autocast, GradScaler
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
        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.loss = loss()
        self.losses = []

    def train(self, epochs: int, log_interval: int = 10, max_steps: int = None) -> list[float]:
        """
        Executes the training loop for a specified number of epochs.

        Args:
            epochs (int): The number of complete passes through the dataset.
            log_interval (int): How often to print the loss. Defaults to 10.
            max_steps (int, optional): Maximum number of steps to train for.

        Returns:
            list[float]: A list of recorded loss values.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        losses = []
        global_step = 0

        print(f"{'Epoch':^6} | {'Step':^8} | {'Loss':^10}")
        print("-" * 30)

        for epoch in range(epochs):
            for step, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)  # (B, T, V)
                    loss = self.loss(logits, y)
                    loss_val = loss.item() # Capture before sealing/accumulation adjustment for logging? 
                                            # Usually you want the actual loss, but `loss` here is scaled? 
                                            # No, loss is just the tensor.
                                            # We divide by grad_accum_steps for backward, but for logging we usually want the "real" average loss.
                                            # So `loss.item()` is valid for the batch.
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    global_step += 1

                    if global_step % log_interval == 0:
                        losses.append(loss_val)
                        print(f"{epoch+1:^6} | {global_step:^8} | {loss_val:^10.4f}")

                    if max_steps and global_step >= max_steps:
                        print("-" * 30)
                        return losses
        
        print("-" * 30)
        self.losses = losses
        return losses

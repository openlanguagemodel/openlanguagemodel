import torch
import torch.nn as nn

import olm.train.callbacks.checkpoint_cb as checkpoint_module
import olm.logging.wandb_logger as wandb_module
from olm.train.callbacks.checkpoint_cb import CheckpointCallback


class DummyTrainer:
    current_epoch = 0
    scheduler = None
    losses = []

    def __init__(self):
        self.model = nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.training_state = {}


def test_checkpoint_callback_skips_non_main_distributed_ranks(monkeypatch, tmp_path):
    callback = CheckpointCallback(str(tmp_path), save_every=1)

    monkeypatch.setattr(checkpoint_module, "is_distributed", lambda: True)
    monkeypatch.setattr(checkpoint_module, "is_main_process", lambda: False)
    monkeypatch.setattr(
        checkpoint_module.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-main rank should not write checkpoints")
        ),
    )

    callback._save_checkpoint(DummyTrainer(), step=1, is_regular=True)


def test_checkpoint_callback_uses_fsdp_trainer_checkpoint_path(tmp_path):
    callback = CheckpointCallback(str(tmp_path), save_every=1)

    class FSDPTrainer:
        current_epoch = 0
        training_state = {}

        def __init__(self):
            self.saved_paths = []

        def save_checkpoint(self, path):
            self.saved_paths.append(path)

    trainer = FSDPTrainer()

    callback._save_checkpoint(trainer, step=3, is_regular=True)

    assert trainer.saved_paths == [str(tmp_path / "step_3.pt")]


def test_wandb_callback_prediction_method_is_not_shadowed(monkeypatch):
    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *items):
            self.rows.append(items)

    class FakeWandB:
        Table = FakeTable

    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(wandb_module, "wandb", FakeWandB)

    callback = wandb_module.WandBCallback(project="test", log_predictions=True)
    callback._should_log = True
    callback.prediction_table = FakeTable(
        ["step", "input_text", "prediction", "target"]
    )

    callback.log_predictions(
        step=1,
        inputs=["hello"],
        predictions=["world"],
        targets=["target"],
    )

    assert callback.prediction_table.rows == [(1, "hello", "world", "target")]

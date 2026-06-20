import importlib

import torch.nn as nn

from olm.train.device import DeviceConfig, TrainerStrategy


def test_auto_trainer_cpu_auto_does_not_reselect_cuda(monkeypatch):
    auto_trainer_module = importlib.import_module("olm.train.trainer.auto_trainer")
    captured = {}
    config = DeviceConfig(
        num_gpus=1,
        num_cpus=1,
        cuda_available=True,
        strategy=TrainerStrategy.SINGLE_CPU,
        device_type="cpu",
    )

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    def fail_determine_strategy(*args, **kwargs):
        raise AssertionError("cpu:auto should not be re-determined")

    monkeypatch.setattr(
        auto_trainer_module,
        "parse_device_string",
        lambda device, model=None: config,
    )
    monkeypatch.setattr(
        auto_trainer_module, "determine_strategy", fail_determine_strategy
    )
    monkeypatch.setattr(auto_trainer_module, "Trainer", FakeTrainer)

    auto_trainer_module.AutoTrainer(
        nn.Linear(2, 2),
        object(),
        object(),
        device="cpu:auto",
        verbose=False,
    )

    assert captured["device"] == "cpu"
    assert captured["use_amp"] is False


def test_auto_trainer_preserves_explicit_cuda_device(monkeypatch):
    auto_trainer_module = importlib.import_module("olm.train.trainer.auto_trainer")
    captured = {}
    config = DeviceConfig(
        num_gpus=1,
        num_cpus=1,
        cuda_available=True,
        strategy=TrainerStrategy.SINGLE_GPU,
        device_type="cuda",
    )

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        auto_trainer_module,
        "parse_device_string",
        lambda device, model=None: config,
    )
    monkeypatch.setattr(auto_trainer_module, "is_distributed", lambda: False)
    monkeypatch.setattr(auto_trainer_module, "Trainer", FakeTrainer)

    auto_trainer_module.AutoTrainer(
        nn.Linear(2, 2),
        object(),
        object(),
        device="cuda:0",
        verbose=False,
    )

    assert captured["device"] == "cuda:0"

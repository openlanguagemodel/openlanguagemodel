import math
import os
import random
import re
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import IterableDataset, TensorDataset

from olm.core import (
    get_rng_state,
    make_generator,
    seed_everything,
    seed_worker,
    set_rng_state,
)
from olm.data.datasets import DataLoader
from olm.nn.blocks import LM
from olm.train import (
    AdamW,
    CheckpointCallback,
    CrossEntropyLoss,
    DDPTrainer,
    EarlyStoppingCallback,
    LMOutput,
    MaskedCELoss,
    ThroughputCallback,
    Trainer,
    TrainerCallback,
    ValidationCallback,
    find_latest_checkpoint,
    read_checkpoint,
    unwrap_model,
)

VOCAB_SIZE = 64
CONTEXT_LENGTH = 8


class ShardedStream(IterableDataset):
    """An iterable dataset that splits its samples across DataLoader workers."""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for index, (x, y) in enumerate(zip(self.input_ids, self.labels)):
            if worker_info is None or index % worker_info.num_workers == worker_info.id:
                yield x, y


class ScriptedValidation(TrainerCallback):
    """Reports a fixed sequence of validation losses, like ValidationCallback."""

    order = -10

    def __init__(self, values, eval_every):
        self.values = list(values)
        self.eval_every = eval_every

    def on_step_end(self, trainer, step, loss):
        if step % self.eval_every == 0 and self.values:
            trainer.training_state["val_loss"] = self.values.pop(0)
            trainer.training_state["val_step"] = step


def _pairs(samples, seed=123):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, VOCAB_SIZE, (samples, CONTEXT_LENGTH), generator=generator
    )
    return input_ids, torch.roll(input_ids, shifts=-1, dims=1)


def _tiny_lm(dropout=0.1):
    return LM(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=CONTEXT_LENGTH,
        dropout=dropout,
    )


def _loader(samples=18, num_workers=0, iterable=False, batch_size=2, **kwargs):
    input_ids, labels = _pairs(samples)
    dataset = (
        ShardedStream(input_ids, labels)
        if iterable
        else TensorDataset(input_ids, labels)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        **kwargs,
    )


def _resumable_run(checkpoint_dir, resume_from=None, callbacks=(), **loader_kwargs):
    """Three epochs of 9 batches with dropout, shuffling and grad accumulation."""
    seed_everything(0)
    trainer = Trainer(
        _tiny_lm(),
        AdamW,
        _loader(**loader_kwargs),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        grad_accum_steps=2,
        use_amp=False,
        learning_rate=1e-2,
        total_steps=15,
        callbacks=[
            CheckpointCallback(str(checkpoint_dir), save_every=1, keep_last_n=20),
            *callbacks,
        ],
        resume_from=resume_from,
    )
    trainer.train(epochs=3, log_interval=1000)
    return trainer


@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {},
        {"num_workers": 2},
        {"generator": "own", "worker_init_fn": seed_worker},
        {"iterable": True},
    ],
    ids=["shuffled", "persistent-workers", "own-generator", "iterable"],
)
def test_resumed_run_reproduces_uninterrupted_run(tmp_path, loader_kwargs):
    def run(name, resume_from=None):
        kwargs = dict(loader_kwargs)
        if kwargs.get("generator") == "own":
            kwargs["generator"] = make_generator(7)
        return _resumable_run(tmp_path / name, resume_from=resume_from, **kwargs)

    reference = run("full")
    assert reference.global_step == 15  # 3 epochs x ceil(9 batches / 2)

    # Step 3 is inside the first epoch; step 7 is in the second, after an epoch
    # boundary and a partial accumulation group.
    for step in (3, 7):
        resumed = run(f"resumed_{step}", tmp_path / "full" / f"step_{step}.pt")

        assert resumed.global_step == reference.global_step
        assert resumed.losses == reference.losses
        assert resumed.total_tokens_processed == reference.total_tokens_processed
        assert resumed.scheduler.last_epoch == reference.scheduler.last_epoch
        assert (
            resumed.optimizer.param_groups[0]["lr"]
            == reference.optimizer.param_groups[0]["lr"]
        )
        for resumed_param, reference_param in zip(
            resumed.model.parameters(), reference.model.parameters()
        ):
            assert torch.equal(resumed_param, reference_param)


def test_validation_leaves_the_training_random_state_alone(tmp_path):
    # A validation loader with persistent workers draws a worker seed from the
    # global generator only for its first iterator in a process. Unless
    # validation restores the random state, a resumed process draws once more
    # than the uninterrupted one, which shifts dropout and shuffling.
    def run(name, resume_from=None):
        val_input_ids, val_labels = _pairs(4, seed=9)
        validation = ValidationCallback(
            DataLoader(
                TensorDataset(val_input_ids, val_labels),
                batch_size=2,
                num_workers=2,
                pin_memory=False,
            ),
            eval_every=2,
        )
        return _resumable_run(
            tmp_path / name, resume_from=resume_from, callbacks=[validation]
        )

    reference = run("full")
    for step in (3, 7):
        resumed = run(f"resumed_{step}", tmp_path / "full" / f"step_{step}.pt")
        assert resumed.losses == reference.losses

    # Validating does not change the training run at all.
    assert _resumable_run(tmp_path / "unvalidated").losses == reference.losses


def test_load_checkpoint_rolls_back_a_trainer_with_persistent_workers(tmp_path):
    trainer = _resumable_run(tmp_path, num_workers=2)
    reference = list(trainer.losses)

    # The loader now holds the persistent iterator of the last epoch. Step 2 is
    # in the first epoch, whose iterator the interrupted run created afresh;
    # steps 5 and 7 continue in the second epoch, which reset it.
    for step in (2, 5, 7):
        trainer.load_checkpoint(tmp_path / f"step_{step}.pt")
        trainer.train(epochs=3, log_interval=1000)
        assert trainer.losses == reference


def test_resume_restores_the_lazily_created_default_scheduler(tmp_path):
    seed_everything(0)
    trainer = Trainer(
        _tiny_lm(),
        AdamW,
        _loader(samples=16),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        learning_rate=1e-2,
        callbacks=[CheckpointCallback(str(tmp_path), save_every=5)],
    )
    trainer.train(epochs=1, log_interval=1000)
    saved_lr = read_checkpoint(tmp_path / "step_5.pt")["optimizer"]["param_groups"][0][
        "lr"
    ]

    resumed = Trainer(
        _tiny_lm(),
        AdamW,
        _loader(samples=16),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        learning_rate=1e-2,
        resume_from=tmp_path / "step_5.pt",
    )

    # The default scheduler used to exist only inside train(), so the old
    # resume recipe silently skipped it and the learning rate warmed up again.
    assert resumed.scheduler is not None
    assert resumed.scheduler.last_epoch == 5
    assert resumed.total_steps == trainer.total_steps == 8
    assert resumed.optimizer.param_groups[0]["lr"] == saved_lr
    assert 0.0 < saved_lr < 1e-2  # mid-way down the cosine


def test_resume_from_latest_and_directories(tmp_path, capsys):
    def make_trainer(resume_from, checkpoint_dir=tmp_path):
        return Trainer(
            _tiny_lm(dropout=0.0),
            AdamW,
            _loader(samples=8),
            device="cpu",
            context_length=CONTEXT_LENGTH,
            use_amp=False,
            use_warmup_cosine=False,
            callbacks=[CheckpointCallback(str(checkpoint_dir), save_every=2)],
            resume_from=resume_from,
        )

    fresh = make_trainer("latest")
    assert fresh.global_step == 0
    assert "starting a new run" in capsys.readouterr().out

    fresh.train(epochs=1, max_steps=3, log_interval=1000)
    # step_2.pt from save_every and step_3.pt from the final save.
    assert find_latest_checkpoint(tmp_path).name == "step_3.pt"

    assert make_trainer("latest").global_step == 3
    assert make_trainer(tmp_path).global_step == 3
    assert make_trainer(tmp_path / "step_2.pt").global_step == 2

    with pytest.raises(FileNotFoundError):
        make_trainer(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="CheckpointCallback"):
        Trainer(
            _tiny_lm(),
            AdamW,
            _loader(samples=8),
            device="cpu",
            context_length=CONTEXT_LENGTH,
            resume_from="latest",
        )


def test_checkpoint_format_loads_with_weights_only(tmp_path):
    seed_everything(0)
    trainer = Trainer(
        _tiny_lm(),
        AdamW,
        _loader(samples=8),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
    )
    trainer.train(epochs=1, max_steps=2, log_interval=1000)
    trainer.save_checkpoint(tmp_path / "ckpt.pt")

    checkpoint = read_checkpoint(tmp_path / "ckpt.pt")  # weights_only=True

    assert checkpoint["format_version"] == 1
    assert checkpoint["trainer"]["global_step"] == 2
    assert checkpoint["trainer"]["batches_in_epoch"] == 2
    assert checkpoint["scheduler_class"] == "WarmupCosineScheduler"
    assert len(checkpoint["rng"]) == 1
    # Pre-version-1 names stay available for existing scripts.
    assert checkpoint["model_state_dict"] is checkpoint["model"]
    assert checkpoint["step"] == 2

    model = _tiny_lm()
    model.load_state_dict(checkpoint["model"])
    for loaded, trained in zip(model.parameters(), trainer.model.parameters()):
        assert torch.equal(loaded, trained)


def test_legacy_checkpoint_callback_files_still_resume(tmp_path):
    seed_everything(0)
    old_run = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=16),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        grad_accum_steps=2,
        use_amp=False,
        use_warmup_cosine=False,
    )
    old_run.train(epochs=1, max_steps=3, log_interval=1000)
    # The pre-version-1 CheckpointCallback layout, as a DDP run wrote it.
    torch.save(
        {
            "step": 3,
            "epoch": 0,
            "model_state_dict": {
                f"module.{key}": value
                for key, value in old_run.model.state_dict().items()
            },
            "optimizer_state_dict": old_run.optimizer.state_dict(),
            "scaler_state_dict": {},
            "losses": old_run.losses,
        },
        tmp_path / "step_3.pt",
    )

    with pytest.warns(UserWarning, match="predates checkpoint format version 1"):
        trainer = Trainer(
            _tiny_lm(dropout=0.0),
            AdamW,
            _loader(samples=16),
            device="cpu",
            context_length=CONTEXT_LENGTH,
            grad_accum_steps=2,
            use_amp=False,
            use_warmup_cosine=False,
            resume_from=tmp_path / "step_3.pt",
        )

    assert trainer.global_step == 3
    assert trainer.batches_in_epoch == 6  # 3 steps x 2 accumulated batches
    assert trainer.losses == old_run.losses
    for loaded, saved in zip(trainer.model.parameters(), old_run.model.parameters()):
        assert torch.equal(loaded, saved)

    trainer.train(epochs=1, log_interval=1000)
    assert trainer.global_step == 4  # 8 batches in the epoch, 6 already used


def test_checkpoints_hold_unwrapped_model_keys(tmp_path):
    model = _tiny_lm(dropout=0.0)
    # torch.compile only wraps the module here; nothing compiles until a call.
    wrapped = torch.compile(nn.DataParallel(model))
    assert unwrap_model(wrapped) is model

    trainer = Trainer(
        wrapped,
        AdamW,
        _loader(samples=4),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
    )
    trainer.save_checkpoint(tmp_path / "ckpt.pt")

    saved = read_checkpoint(tmp_path / "ckpt.pt")["model"]
    assert set(saved) == set(model.state_dict())


def test_legacy_fsdp_layout_reads_model_weights(tmp_path):
    model = nn.Linear(2, 2)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": None,
            "global_step": 7,
            "epoch": 1,
        },
        tmp_path / "fsdp.pt",
    )

    with pytest.warns(UserWarning, match="rank 0's shard"):
        checkpoint = read_checkpoint(tmp_path / "fsdp.pt")

    assert checkpoint["format_version"] == 0
    assert checkpoint["optimizer"] is None
    assert checkpoint["trainer"]["global_step"] == 7
    assert torch.equal(checkpoint["model"]["weight"], model.weight)


@pytest.mark.parametrize(
    "grad_accum_steps, max_steps, steps_per_epoch, expected",
    [
        (1, None, None, 8),
        (4, None, None, 2),
        (3, None, None, 3),
        (1, 3, None, 3),
        (1, None, 5, 5),
    ],
)
def test_default_schedule_counts_optimizer_steps(
    grad_accum_steps, max_steps, steps_per_epoch, expected
):
    trainer = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=16),  # 8 batches
        device="cpu",
        context_length=CONTEXT_LENGTH,
        grad_accum_steps=grad_accum_steps,
        use_amp=False,
        learning_rate=1e-3,
        min_lr=0.0,
    )

    trainer.train(
        epochs=1,
        max_steps=max_steps,
        steps_per_epoch=steps_per_epoch,
        log_interval=1000,
    )

    assert trainer.total_steps == expected
    assert trainer.global_step == expected
    # The schedule reaches its end exactly when training does.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_default_schedule_warns_when_the_dataloader_has_no_length():
    trainer = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=4, iterable=True),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
    )

    with pytest.warns(UserWarning, match="10000 optimizer steps per epoch"):
        trainer.train(epochs=1, log_interval=1000)
    assert trainer.total_steps == 10000


def test_default_schedule_without_a_length_uses_the_shorter_limit():
    trainer = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=12, iterable=True),  # 6 batches per epoch
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        min_lr=0.0,
    )

    trainer.train(epochs=2, steps_per_epoch=3, max_steps=1000, log_interval=1000)

    assert trainer.total_steps == trainer.global_step == 6
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def _early_stopping_trainer(callbacks, samples=40):
    return Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=samples),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        use_warmup_cosine=False,
        callbacks=callbacks,
    )


def test_early_stopping_counts_validations_not_steps():
    # One validation at step 1, then none: patience is never used up. The old
    # callback re-read the stale val_loss every step and stopped at step 4.
    trainer = _early_stopping_trainer(
        [ScriptedValidation([1.0], eval_every=1), EarlyStoppingCallback(patience=3)]
    )
    trainer.train(epochs=1, log_interval=1000)
    assert trainer.global_step == 20


def test_early_stopping_stops_after_patience_validations():
    early_stopping = EarlyStoppingCallback(patience=3)
    # Listed before the validation callback on purpose: ordering must not matter.
    trainer = _early_stopping_trainer(
        [early_stopping, ScriptedValidation([1.0, 0.9, 0.95, 0.96, 0.97], eval_every=2)]
    )
    trainer.train(epochs=1, log_interval=1000)

    # Validations at steps 2, 4, 6, 8, 10; the last three do not improve on 0.9.
    assert trainer.global_step == 10
    assert early_stopping.best_value == 0.9
    assert early_stopping.best_step == 4
    assert early_stopping.stopped_step == 10


def test_early_stopping_min_delta_and_max_mode():
    class State:
        def __init__(self):
            self.training_state = {}

    def feed(callback, values):
        trainer = State()
        for step, value in enumerate(values, start=1):
            trainer.training_state.update(val_acc=value, val_step=step)
            callback.on_step_end(trainer, step, 0.0)
        return trainer.training_state.get("should_stop", False)

    # 0.505 and 0.508 improve by less than min_delta, so they count as waits.
    patient = EarlyStoppingCallback(
        patience=2, min_delta=0.01, monitor="val_acc", mode="max"
    )
    assert feed(patient, [0.5, 0.505, 0.508]) is True
    assert patient.best_value == 0.5

    improving = EarlyStoppingCallback(patience=2, monitor="val_acc", mode="max")
    assert feed(improving, [0.5, 0.6, 0.7, 0.8]) is False
    assert improving.best_value == 0.8

    with pytest.raises(ValueError):
        EarlyStoppingCallback(mode="maximum")


def test_early_stopping_state_survives_a_resume(tmp_path):
    early_stopping = EarlyStoppingCallback(patience=3)
    trainer = _early_stopping_trainer(
        [ScriptedValidation([1.0, 0.9, 0.95], eval_every=2), early_stopping]
    )
    trainer.train(epochs=1, max_steps=6, log_interval=1000)
    trainer.save_checkpoint(tmp_path / "step_6.pt")

    restored = EarlyStoppingCallback(patience=3)
    resumed = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=40),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        use_warmup_cosine=False,
        callbacks=[ScriptedValidation([0.96, 0.97], eval_every=2), restored],
        resume_from=tmp_path / "step_6.pt",
    )
    assert (restored.best_value, restored.wait) == (0.9, 1)

    resumed.train(epochs=1, log_interval=1000)
    assert resumed.global_step == 10  # two more non-improving validations


def test_best_checkpoint_holds_the_weights_of_the_validated_step(tmp_path):
    seed_everything(0)
    val_input_ids, val_labels = _pairs(4, seed=7)
    validation = ValidationCallback(
        DataLoader(
            TensorDataset(val_input_ids, val_labels), batch_size=2, pin_memory=False
        ),
        eval_every=2,
    )
    checkpoints = CheckpointCallback(str(tmp_path), save_every=100)
    trainer = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=16),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        learning_rate=1e-2,
        # CheckpointCallback first: callbacks run by `order`, not list position.
        callbacks=[checkpoints, validation],
    )
    trainer.train(epochs=1, log_interval=1000)

    best_step, best_loss = min(validation.val_losses, key=lambda entry: entry[1])
    best = read_checkpoint(tmp_path / "best_model.pt")
    assert best["trainer"]["global_step"] == best_step
    assert checkpoints.best_step == best_step
    assert checkpoints.best_value == best_loss
    assert best["callbacks"]["CheckpointCallback"]["best_value"] == best_loss


def test_validation_callback_accepts_lm_output_models():
    class StructuredLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = _tiny_lm(dropout=0.0)

        def forward(self, input_ids):
            return LMOutput(logits=self.lm(input_ids), aux_losses=[torch.tensor(0.5)])

    loader = _loader(samples=4)
    validation = ValidationCallback(loader, eval_every=1)
    trainer = Trainer(
        StructuredLM(),
        AdamW,
        loader,
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        callbacks=[validation],
    )
    trainer.train(epochs=1, log_interval=1000)

    assert [step for step, _ in validation.val_losses] == [1, 2]
    assert trainer.training_state["val_step"] == 2
    assert all(math.isfinite(loss) for _, loss in validation.val_losses)


def test_masked_ce_loss_trains_without_a_mask():
    logits = torch.randn(2, 5, VOCAB_SIZE)
    labels = torch.randint(0, VOCAB_SIZE, (2, 5))
    labels[0, :2] = -100

    assert torch.allclose(
        MaskedCELoss()(logits, labels), CrossEntropyLoss()(logits, labels)
    )

    # A mask removes more positions; ignored labels never count, even if masked in.
    mask = torch.ones(2, 5)
    mask[1, 3:] = 0
    expected = CrossEntropyLoss()(logits[1:, :3], labels[1:, :3]) * 3 + (
        CrossEntropyLoss()(logits[:1, 2:], labels[:1, 2:]) * 3
    )
    assert torch.allclose(MaskedCELoss()(logits, labels, mask), expected / 6)

    trainer = Trainer(
        _tiny_lm(dropout=0.0),
        AdamW,
        _loader(samples=4),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        use_amp=False,
        loss=MaskedCELoss,
    )
    losses = trainer.train(epochs=1, log_interval=1000)
    assert len(losses) == 2 and all(math.isfinite(loss) for loss in losses)


class _SlowLM(nn.Module):
    """The tiny LM, with a forward pass that takes at least 50 ms."""

    def __init__(self):
        super().__init__()
        self.lm = _tiny_lm(dropout=0.0)

    def forward(self, input_ids):
        time.sleep(0.05)
        return self.lm(input_ids)


def test_throughput_callback_times_the_whole_step():
    throughput = ThroughputCallback(log_every=1)
    trainer = Trainer(
        _SlowLM(),
        AdamW,
        _loader(samples=8),
        device="cpu",
        context_length=CONTEXT_LENGTH,
        grad_accum_steps=2,
        use_amp=False,
        callbacks=[throughput],
    )
    trainer.train(epochs=1, log_interval=1000)

    # Each step runs two 50 ms forward passes over 2 x 2 x 8 = 32 tokens.
    tokens_per_step = 2 * 2 * CONTEXT_LENGTH
    assert trainer.training_state["tokens_per_sec"] <= tokens_per_step / 0.1
    assert trainer.training_state["samples_per_sec"] == pytest.approx(
        trainer.training_state["tokens_per_sec"] / CONTEXT_LENGTH
    )

    with pytest.warns(DeprecationWarning, match="batch_size"):
        ThroughputCallback(batch_size=8)


def test_training_summary_throughput_counts_only_this_call(tmp_path, capsys):
    def make_trainer(resume_from=None):
        return Trainer(
            _SlowLM(),
            AdamW,
            _loader(samples=16),  # 8 steps of one batch
            device="cpu",
            context_length=CONTEXT_LENGTH,
            use_amp=False,
            callbacks=[CheckpointCallback(str(tmp_path), save_every=4)],
            resume_from=resume_from,
        )

    make_trainer().train(epochs=1, log_interval=1000)
    capsys.readouterr()

    make_trainer(tmp_path / "step_4.pt").train(epochs=1, log_interval=1000)
    summary = capsys.readouterr().out
    throughput = float(re.search(r"Average Throughput: (\d+)", summary).group(1))
    # This call ran 4 batches of 2 x 8 tokens, each forward taking 50 ms or
    # more. Counting the 4 restored batches too would about double the value.
    assert throughput <= 4 * 2 * CONTEXT_LENGTH / (4 * 0.05)

    # Resuming a finished run trains nothing, so there is no throughput.
    make_trainer(tmp_path / "step_8.pt").train(epochs=1, log_interval=1000)
    assert "Average Throughput" not in capsys.readouterr().out


def test_callbacks_run_in_order():
    calls = []

    class Recorder(TrainerCallback):
        def __init__(self, name, order):
            self.name = name
            self.order = order

        def on_step_end(self, trainer, step, loss):
            calls.append(self.name)

    trainer = _early_stopping_trainer(
        [Recorder("late", 10), Recorder("default", 0), Recorder("early", -10)],
        samples=2,
    )
    trainer.train(epochs=1, log_interval=1000)
    assert calls == ["early", "default", "late"]


def test_seed_everything_reproduces_initialization_and_training():
    def run(seed):
        seed_everything(seed)
        model = _tiny_lm()
        trainer = Trainer(
            model,
            AdamW,
            _loader(samples=8),
            device="cpu",
            context_length=CONTEXT_LENGTH,
            use_amp=False,
        )
        return [p.detach().clone() for p in model.parameters()], trainer.train(
            epochs=1, log_interval=1000
        )

    params_a, losses_a = run(0)
    params_b, losses_b = run(0)
    params_c, losses_c = run(1)

    assert losses_a == losses_b
    assert all(torch.equal(a, b) for a, b in zip(params_a, params_b))
    assert losses_a != losses_c
    assert seed_everything(5) == 5


def test_rng_state_round_trip():
    seed_everything(3)
    state = get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))

    random.random(), np.random.rand(), torch.rand(1)
    set_rng_state(state)

    assert (random.random(), np.random.rand()) == expected[:2]
    assert torch.equal(torch.rand(1), expected[2])


class _WorkerRandom(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        worker_seed = torch.initial_seed() % 2**32
        return torch.tensor(
            [
                random.random(),
                random.Random(worker_seed).random(),
                float(np.random.rand()),
            ],
            dtype=torch.float64,
        )


def test_seed_worker_and_make_generator_make_workers_reproducible():
    def draw(seed):
        loader = torch.utils.data.DataLoader(
            _WorkerRandom(),
            batch_size=1,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )
        return torch.cat(list(loader))

    first = draw(0)
    # Batches 0 and 1 are each worker's first draw: seed_worker seeded
    # Python's generator from the worker's torch seed.
    assert torch.equal(first[:2, 0], first[:2, 1])
    assert torch.equal(first, draw(0))
    assert not torch.equal(first[0], first[1])  # the two workers differ
    assert not torch.equal(first, draw(1))


def test_seed_everything_deterministic_mode(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]


def _ddp_worker(rank, world_size, rendezvous, checkpoint_dir, resume_from, output):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        seed_everything(0)
        input_ids, labels = _pairs(36)
        loader = DataLoader(
            TensorDataset(input_ids, labels),
            batch_size=2,
            pin_memory=False,
            distributed=True,
        )
        trainer = DDPTrainer(
            _tiny_lm(),
            AdamW,
            loader,
            device="cpu",
            context_length=CONTEXT_LENGTH,
            grad_accum_steps=2,
            use_amp=False,
            learning_rate=1e-2,
            callbacks=[CheckpointCallback(checkpoint_dir, save_every=2)],
            resume_from=resume_from,
        )
        trainer.train(epochs=2, log_interval=1000)
        torch.save(
            {"losses": trainer.losses, "model": trainer.model.module.state_dict()},
            f"{output}.rank{rank}",
        )
    finally:
        dist.destroy_process_group()


def test_ddp_resumed_run_reproduces_uninterrupted_run(tmp_path):
    def launch(name, resume_from=None):
        mp.spawn(
            _ddp_worker,
            args=(
                2,
                str(tmp_path / f"rendezvous_{name}"),
                str(tmp_path / f"checkpoints_{name}"),
                resume_from,
                str(tmp_path / name),
            ),
            nprocs=2,
        )
        return [torch.load(tmp_path / f"{name}.rank{rank}") for rank in range(2)]

    reference = launch("full")
    # Step 6 is in the second epoch, so the DistributedSampler has reshuffled.
    resumed = launch(
        "resumed", resume_from=str(tmp_path / "checkpoints_full" / "step_6.pt")
    )

    assert len(reference[0]["losses"]) == 10  # 2 epochs x ceil(9 batches / 2) per rank
    for rank in range(2):
        assert resumed[rank]["losses"] == reference[rank]["losses"]
        for key, value in reference[rank]["model"].items():
            assert torch.equal(resumed[rank]["model"][key], value)

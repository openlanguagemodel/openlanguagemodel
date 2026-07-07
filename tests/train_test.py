import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, TensorDataset

from olm.data.datasets import DataLoader
from olm.nn.blocks import LM
from olm.train import LMOutput, Trainer
from olm.train.callbacks import ValidationCallback
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.mtp import MTPLoss
from olm.train.optim import AdamW
from olm.train.trainer import TrainerCallback


class FiniteIterablePairs(IterableDataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __iter__(self):
        for x, y in zip(self.input_ids, self.labels):
            yield x, y


class StructuredOutputToyModel(nn.Module):
    def __init__(self, vocab_size=16, embed_dim=8, output_mode="lm_output"):
        super().__init__()
        self.output_mode = output_mode
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)
        self.aux_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, input_ids):
        logits = self.proj(self.embed(input_ids))
        aux_loss = self.aux_scale.square()
        if self.output_mode == "dict":
            return {"logits": logits, "aux_losses": {"toy": aux_loss}}
        return LMOutput(logits=logits, aux_losses=[aux_loss])


class MTPOutputToyModel(nn.Module):
    def __init__(self, vocab_size=16, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.main = nn.Linear(embed_dim, vocab_size)
        self.future = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        logits = self.main(hidden)
        mtp_logits = [self.future(hidden[:, :-2])]
        return LMOutput(logits=logits, mtp_logits=mtp_logits)


def _toy_loader(vocab_size=16, context_length=6, samples=4):
    input_ids = torch.randint(0, vocab_size, (samples, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    return DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def test_trainer_runs_tiny_lm_for_two_steps():
    torch.manual_seed(0)

    vocab_size = 64
    context_length = 16
    samples = 8

    input_ids = torch.randint(0, vocab_size, (samples, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)

    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, max_steps=2, log_interval=100)

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))


def test_trainer_accepts_lm_output_with_aux_losses():
    torch.manual_seed(0)
    loader = _toy_loader()
    model = StructuredOutputToyModel(output_mode="lm_output")

    trainer = Trainer(
        model,
        torch.optim.AdamW,
        loader,
        device="cpu",
        context_length=6,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, max_steps=1, log_interval=100)

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_accepts_dict_output_with_aux_losses():
    torch.manual_seed(0)
    loader = _toy_loader()
    model = StructuredOutputToyModel(output_mode="dict")

    trainer = Trainer(
        model,
        torch.optim.AdamW,
        loader,
        device="cpu",
        context_length=6,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, max_steps=1, log_interval=100)

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_can_add_mtp_loss_from_structured_output():
    torch.manual_seed(0)
    loader = _toy_loader()
    model = MTPOutputToyModel()

    trainer = Trainer(
        model,
        torch.optim.AdamW,
        loader,
        device="cpu",
        context_length=6,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
        mtp_loss=MTPLoss(num_heads=1),
    )

    losses = trainer.train(epochs=1, max_steps=1, log_interval=100)

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_steps_on_partial_gradient_accumulation():
    torch.manual_seed(0)

    vocab_size = 64
    context_length = 8
    samples = 6

    input_ids = torch.randint(0, vocab_size, (samples, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        grad_accum_steps=4,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, log_interval=100)

    assert len(losses) == 1
    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_flushes_partial_accumulation_for_finite_iterables():
    torch.manual_seed(0)

    vocab_size = 64
    context_length = 8
    samples = 6

    input_ids = torch.randint(0, vocab_size, (samples, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        FiniteIterablePairs(input_ids, labels),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        grad_accum_steps=4,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, log_interval=100)

    assert len(losses) == 1
    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_warmup_cosine_handles_single_step_run():
    torch.manual_seed(0)

    vocab_size = 32
    context_length = 8
    input_ids = torch.randint(0, vocab_size, (2, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        use_amp=False,
        learning_rate=1e-3,
        use_warmup_cosine=True,
    )

    losses = trainer.train(epochs=1, max_steps=1, log_interval=100)

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_trainer_honors_callback_stop_request():
    class StopAfterFirstStep(TrainerCallback):
        def on_step_end(self, trainer, step: int, loss: float) -> None:
            trainer.training_state["should_stop"] = True

    torch.manual_seed(0)

    vocab_size = 32
    context_length = 8
    input_ids = torch.randint(0, vocab_size, (8, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        use_amp=False,
        callbacks=[StopAfterFirstStep()],
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    losses = trainer.train(epochs=1, max_steps=4, log_interval=100)

    assert len(losses) == 1
    assert trainer.global_step == 1


def test_validation_callback_uses_trainer_device_on_cpu():
    torch.manual_seed(0)

    vocab_size = 64
    context_length = 8
    input_ids = torch.randint(0, vocab_size, (4, context_length))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = LM(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        max_seq_len=context_length,
        dropout=0.0,
    )
    validation = ValidationCallback(loader, eval_every=1, device=None, use_amp=True)

    trainer = Trainer(
        model,
        AdamW,
        loader,
        device="cpu",
        context_length=context_length,
        use_amp=False,
        callbacks=[validation],
        learning_rate=1e-3,
        use_warmup_cosine=False,
    )

    trainer.train(epochs=1, max_steps=1, log_interval=100)

    assert len(validation.val_losses) == 1
    assert torch.isfinite(torch.tensor(validation.val_losses[0][1]))


def test_cross_entropy_accepts_non_contiguous_logits():
    logits = torch.randn(2, 3, 16).transpose(0, 1)
    labels = torch.randint(0, 16, (3, 2))

    loss = CrossEntropyLoss()(logits, labels)

    assert torch.isfinite(loss)

# Experiment Tracking with Weights & Biases

OLM integrates [Weights & Biases](https://wandb.ai/) (wandb) through a single callback. Add it to any trainer to get live charts of loss, perplexity, learning rate, throughput, and system metrics, plus optional gradient histograms, checkpoint artifacts, alerts, and hyperparameter sweeps.

Install the extra and authenticate:

```bash
pip install "openlanguagemodel[wandb]"
wandb login
```

If wandb is not installed, OLM degrades gracefully and the rest of the library works unchanged.

## Basic usage

Construct a [`WandBCallback`](../api/logging.md#wandbcallback) and pass it to the trainer. It automatically captures your model size, optimizer, and training configuration.

```python
from olm.logging import WandBCallback
from olm.train import Trainer

wandb_cb = WandBCallback(
    project="my-language-model",
    name="gpt2-baseline",
    tags=["gpt2", "fineweb"],
)

trainer = Trainer(
    model, optimizer, loader,
    device="cuda", context_length=1024,
    callbacks=[wandb_cb],
)
trainer.train(epochs=1, max_steps=10_000)
```

This logs, every step: `train/loss`, `train/perplexity`, `train/learning_rate`, `train/tokens_per_sec`, and GPU/CPU statistics.

## Gradient and weight tracking

Watch for vanishing or exploding gradients with histograms:

```python
wandb_cb = WandBCallback(
    project="my-language-model",
    log_gradients=True,     # gradient histograms
    watch_model=True,       # wandb.watch() for detailed tracking
    watch_freq=1000,
)
```

## Checkpoint artifacts

Version your model checkpoints as wandb artifacts so every run is reproducible:

```python
wandb_cb = WandBCallback(project="my-language-model", log_model=True)
```

## Alerts

Be notified when a metric crosses a threshold — for example, if the loss diverges:

```python
wandb_cb = WandBCallback(
    project="my-language-model",
    alert_thresholds={
        "train/loss": {"max": 10.0},           # alert if loss climbs above 10
        "train/learning_rate": {"min": 1e-6},  # alert if the LR collapses
    },
)
```

## Hyperparameter sweeps

OLM provides helpers to launch Bayesian (or grid/random) sweeps. Start from the template and customize it:

```python
from olm.logging import create_sweep, get_sweep_config_template
import wandb

config = get_sweep_config_template("bayes")
config["parameters"]["learning_rate"] = {"min": 1e-5, "max": 1e-3}
config["parameters"]["batch_size"] = {"values": [16, 32, 64]}

sweep_id = create_sweep(config, project="my-language-model")

def train_run():
    wandb.init()
    cfg = wandb.config
    # build the trainer using cfg.learning_rate, cfg.batch_size, ...
    # trainer.train(...)

wandb.agent(sweep_id, function=train_run, count=20)
```

## Offline mode

For air-gapped machines, log locally and sync later with `wandb sync`:

```python
wandb_cb = WandBCallback(project="my-language-model", offline=True)
```

## Distributed training

The callback is distributed-aware: only rank 0 logs, so there are no duplicate runs. Add it to a `DDPTrainer` or `FSDPTrainer` exactly as you would to a single-GPU `Trainer`:

```python
from olm.train.trainer import DDPTrainer

trainer = DDPTrainer(
    model, torch.optim.AdamW, loader,
    device=device, context_length=1024,
    callbacks=[WandBCallback(project="distributed-run")],
)
```

## Next steps

- See [`examples/wandb_example.py`](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/examples/wandb_example.py) for a complete, runnable script.
- Browse the full [`WandBCallback` API](../api/logging.md) for every option.

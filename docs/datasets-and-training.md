<h2> Datasets and Training </h2>

The `olm` library is designed to handle massive amounts of text data without using up all your computer's memory. It does this by "streaming" the data—reading it bit by bit as the model needs it, rather than loading everything at once. This allows you to train on datasets that are much larger than your hard drive or RAM.

---

<h3>1. Preparing Your Data</h3>

To start training, you first need to tell the library where your text is. We have three main ways to do this:

- **From Local Files**: If you have a folder full of `.txt` files, use `LocalTextDataset`. It scans the directory and streams each file one by one.
- **From Hugging Face**: If you want to use a dataset from the web (like Wikipedia or Common Crawl), use `HuggingFaceTextDataset`. It downloads chunks of data as you train.
- **FineWeb Edu**: A built-in shortcut for a high-quality educational dataset, pre-configured with the best settings.

**Example Usage:**

```python
from olm.data.datasets import LocalTextDataset, FineWebEduDataset

# 1. Loading from your own folder
dataset = LocalTextDataset(
    location="./my_text_folder",
    tokenizer=tk,
    context_length=1024,
    shuffle=True
)

# 2. Or use the built-in FineWeb shortcut
dataset = FineWebEduDataset(
    tokenizer=tk,
    subset="sample-10BT",
    context_length=2048
)
```

**What is Shuffling?**
Shuffling mixes up your data so the model doesn't see the same examples in the same order every time. This is crucial for making the model learn general patterns rather than just memorizing the order of your files.

> [!TIP]
> **Advanced: Shuffling & Sharding**
> For **local files**, we mix the order of the file names. For **web datasets**, we keep a "buffer" of streaming text and shuffle that buffer (default size: 10,000).
>
> If you use multiple GPUs or workers, the library automatically handles **sharding**: it assigns specific pieces of the dataset to each worker so they never process the same data at the same time.

---

<h3>2. The Data Loader</h3>

The `DataLoader` is the bridge between your dataset and your training loop. It handles the heavy lifting of gathering data into "batches" (groups of examples) and moving them to your GPU efficiently.

```python
from olm.data.datasets import DataLoader

# This creates batches of 32 examples and uses 4 CPU cores to prepare data in parallel
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

> [!NOTE]
> **Advanced: Under the Hood**
> The `olm` loader is tuned for high-throughput. It uses `persistent_workers=True` to avoid the "startup lag" between training epochs, and `pin_memory=True` to speed up the transfer of data from your RAM to your GPU.

---

<h3>3. Training Your Model</h3>

The `Trainer` is the "brain" of the library. It manages the actual math and the complicated logic of the training loop.

**Step 1: The Optimizer**
The optimizer is what actually updates the model's weights to make it better. The Trainer is smart—it knows which parts of the model need extra care (like layers that need "weight decay") and which parts don't (like "biases").

```python
from olm.train.optim import AdamW

# You can just pass the class name, and the Trainer handles the parameter grouping for you
trainer = Trainer(
    model=model,
    optimizer=AdamW,
    learning_rate=3e-4,
    weight_decay=0.1,
    ...
)
```

> [!TIP]
> **Advanced: Parameter Grouping**
> The trainer's `_configure_optimizer` logic automatically excludes 1D parameters (like LayerNorm weights and biases) from weight decay, as decaying these often hurts performance.

**Step 2: Scheduling (Warmup)**
Models are like athletes—they need to warm up. The Trainer automatically starts with a very low learning rate and slowly increases it (**warmup**) before gently decreasing it (**cosine decay**). This keeps training stable and prevents the model from "tripping" at the very start.

**Step 3: Pro Training Features**
The Trainer comes with several "pro" features enabled by default:

- **Mixed Precision (AMP)**: Uses specialized hardware on your GPU to make training 2-3x faster.
- **Gradient Accumulation**: If your GPU is too small for a big batch, this trick lets you simulate a big batch by doing several small steps and only updating the model once at the end.
- **Gradient Clipping**: Prevents the model's math from "exploding" if it sees a very strange piece of data.

```python
# A typical training setup
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=loader,
    device="cuda",
    context_length=1024,
    grad_accum_steps=8, # Accumulate for 8 steps to simulate a 8x larger batch
    use_amp=True        # Faster training on modern GPUs
)

# Start training!
trainer.train(epochs=1, log_interval=10)
```

---

<h3>4. Customizing with Callbacks</h3>

Callbacks are like "plugins" for your training. They let you inject your own code at specific moments—like saving the model every hour, or printing a custom message.

**Example: A Simple Progress Printer**

```python
from olm.train.trainer import TrainerCallback

class MyLogger(TrainerCallback):
    def on_step_end(self, trainer, step, loss):
        # This code runs AFTER every optimization step
        if step % 100 == 0:
            print(f"Step {step}: The current loss is {loss:.4f}")

# Just add it to the trainer's list
trainer = Trainer(..., callbacks=[MyLogger()])
```

> [!IMPORTANT]
> **Advanced: Callback Hooks**

---

<h3>5. Saving and Loading</h3>

Once you've trained your model, you'll want to save it to disk for later use. The `olm` library simplifies this by allowing you to save the model and its associated tokenizer together in one directory.

**Saving Your Model**

All models built using the `Block` system (including the `LM` class) have a built-in `.save()` method. You can optionally pass a tokenizer to save it alongside the model.

```python
# Save the model and the tokenizer to a folder
model.save("./checkpoints/final_model", tokenizer=tk)
```

**Loading Your Model**

To load a saved model, use the `load_model` function. It automatically detects if a tokenizer was saved in the same folder and will return both objects if found.

```python
from olm.nn.structure import load_model

# If a tokenizer was saved with the model:
model, tokenizer = load_model("./checkpoints/final_model")

# If only the model was saved:
model = load_model("./checkpoints/no_tokenizer_model")
```

> [!NOTE]
> **Architecture Preservation**
> The `.save()` method preserves the entire model object. This means you don't need to manually define the model's configuration (like `vocab_size` or `num_layers`) when loading; the library reconstructs the exact architecture for you.

---

<h3>6. Experiment Tracking with Weights & Biases</h3>

Weights & Biases (wandb) provides powerful experiment tracking, visualization, and collaboration features for your training runs. The `olm` library includes comprehensive wandb integration that's completely optional and configurable.

**Installation**

To use wandb features, install the library with wandb support:

```bash
pip install openlanguagemodel[wandb]
```

Then authenticate with your wandb account:

```bash
wandb login
```

**Basic Usage**

Add the `WandBCallback` to your trainer to automatically log metrics, hyperparameters, system stats, and more:

```python
from olm.logging import WandBCallback

# Create the callback with your project name
wandb_callback = WandBCallback(
    project="my-language-model",
    name="gpt2-training-run",
    config={"model": "gpt2", "dataset": "fineweb-edu"}
)

# Add it to your trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=loader,
    callbacks=[wandb_callback],
    ...
)

trainer.train(epochs=1)
```

This automatically logs:

- Training metrics (loss, perplexity, learning rate, throughput)
- Hyperparameters and configuration
- System metrics (GPU memory, CPU usage)
- Model gradients and weights (optional)

**Advanced Features**

**1. Gradient and Weight Tracking**

Monitor your model's gradients and weights with histograms:

```python
wandb_callback = WandBCallback(
    project="my-project",
    log_gradients=True,      # Log gradient histograms
    gradient_log_freq=100,   # Log every 100 steps
    watch_model=True         # Use wandb.watch() for detailed tracking
)
```

**2. Model Checkpoint Artifacts**

Automatically save and version your checkpoints:

```python
wandb_callback = WandBCallback(
    project="my-project",
    log_checkpoints=True,           # Save checkpoints as artifacts
    checkpoint_interval=1000,        # Save every 1000 steps
    checkpoint_dir="./checkpoints"   # Where to save locally
)
```

**3. Alert Integration**

Get notified when metrics cross thresholds:

```python
wandb_callback = WandBCallback(
    project="my-project",
    enable_alerts=True,
    alert_thresholds={
        "loss": 5.0,              # Alert if loss > 5.0
        "gradient_norm": 10.0,    # Alert if gradients explode
    }
)
```

**4. Prediction Table Logging**

Log model predictions for qualitative analysis:

```python
# During training, log predictions periodically
wandb_callback.log_predictions(
    inputs=["The quick brown", "Once upon a time"],
    predictions=["fox jumped over", "there was a"],
    targets=["fox jumped", "there was"],
    step=trainer.step
)
```

**5. Hyperparameter Sweeps**

Run hyperparameter optimization with wandb sweeps:

```python
from olm.logging import create_sweep, get_sweep_config_template

# Get a template configuration
sweep_config = get_sweep_config_template()

# Customize for your needs
sweep_config["parameters"] = {
    "learning_rate": {"min": 1e-5, "max": 1e-3},
    "batch_size": {"values": [16, 32, 64]},
    "weight_decay": {"min": 0.0, "max": 0.3}
}

# Create the sweep
sweep_id = create_sweep(sweep_config, project="my-project")

# Run the sweep (define your train function)
def train():
    wandb.init()
    config = wandb.config

    # Use config.learning_rate, config.batch_size, etc.
    trainer = Trainer(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        ...
    )
    trainer.train()

# Launch sweep agents
wandb.agent(sweep_id, function=train, count=10)
```

**6. Offline Mode**

For air-gapped environments or when internet is unavailable:

```python
wandb_callback = WandBCallback(
    project="my-project",
    offline=True  # Logs stored locally, sync later with `wandb sync`
)
```

**7. Distributed Training Support**

WandB integration automatically works with distributed training—only rank 0 logs to avoid duplicates:

```python
# In your distributed training script
from olm.train.trainer import DDPTrainer
from olm.logging import WandBCallback

wandb_callback = WandBCallback(
    project="distributed-training",
    name=f"ddp-run-{rank}"
)

trainer = DDPTrainer(
    model=model,
    optimizer=optimizer,
    dataloader=loader,
    callbacks=[wandb_callback],  # Only rank 0 will log
    ...
)
```

> [!TIP]
> **Complete Examples**
> See `examples/wandb_example.py` for complete working examples including:
>
> - Basic training with all wandb features
> - Prediction table logging
> - Hyperparameter sweeps with Bayesian optimization
> - Distributed training with wandb

> [!IMPORTANT]
> **Graceful Degradation**
> If wandb is not installed, the library will work normally—wandb features are completely optional. Import errors are handled gracefully with helpful messages.

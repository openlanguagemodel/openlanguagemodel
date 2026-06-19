<h2> Datasets and Training </h2>

The `olm` library is designed to handle massive amounts of text data without using up all your computer's memory. It does this by "streaming" the data—reading it bit by bit as the model needs it, rather than loading everything at once. This allows you to train on datasets that are much larger than your hard drive or RAM.

---

<h3>1. Preparing Your Data</h3>

To start training, you first need to tell the library where your text is. The common paths are:

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
> For **local files**, we mix the order of the file names. For **web datasets**, we keep a buffer of streaming text and shuffle that buffer.
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

<h3>5. Distributed Training (Multi-GPU)</h3>

For large models or faster training, v2.2 supports multiple GPUs on a single machine. OLM provides two approaches using PyTorch's native distributed backends:

**DDP (Distributed Data Parallel)** - Best for models that fit on a single GPU

- Replicates the full model on each GPU
- Synchronizes gradients across GPUs after each backward pass
- Simple and reliable for most use cases

**FSDP (Fully Sharded Data Parallel)** - Best for very large models

- Shards (splits) model parameters across GPUs
- Enables training models larger than single GPU memory
- More memory efficient but slightly more complex

**Launch Command**
Both approaches use `torchrun` to launch multiple processes:

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node launch helpers are planned for v4.
```

**DDP Training Example**

```python
from olm.core.dist import setup_distributed, get_local_rank
from olm.train.trainer import DDPTrainer
from olm.data.datasets import DataLoader

# Initialize distributed environment (auto-detects NCCL for GPU, Gloo for CPU)
setup_distributed()

# Create DataLoader with distributed sampling
loader = DataLoader(
    dataset=dataset,
    batch_size=16,
    num_workers=4,
    distributed=True,  # Automatically creates DistributedSampler
)

# Create DDP trainer
trainer = DDPTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=loader,
    device=f"cuda:{get_local_rank()}",  # Each process uses different GPU
    context_length=1024,
    learning_rate=3e-4,
    grad_accum_steps=4,  # Gradient accumulation works with DDP
)

# Train (metrics are automatically aggregated across GPUs)
trainer.train(epochs=10, log_interval=100)
```

> [!TIP]
> **DDP Best Practices**
>
> - Use `distributed=True` in DataLoader to ensure each GPU sees different data
> - Call `loader.sampler.set_epoch(epoch)` at the start of each epoch for proper shuffling
> - Only rank 0 prints logs and saves checkpoints (automatic in DDPTrainer)
> - Effective batch size = `batch_size × num_gpus × grad_accum_steps`

**FSDP Training Example**

```python
from olm.train.trainer import FSDPTrainer
from olm.core.dist import setup_distributed, get_local_rank

setup_distributed()

trainer = FSDPTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=DataLoader(dataset, batch_size=8, distributed=True),
    device=f"cuda:{get_local_rank()}",
    context_length=2048,
    learning_rate=3e-4,

    # FSDP-specific configuration
    sharding_strategy="FULL_SHARD",  # Full sharding (most memory efficient)
    auto_wrap_policy="size",  # Auto-wrap layers with 100M+ params
    min_num_params=1e8,  # Wrap threshold (default: 100M parameters)
    mixed_precision_policy="bf16",  # BF16 training (faster, requires Ampere+ GPUs)
    cpu_offload=False,  # Set True to offload to CPU (slower but saves memory)
    backward_prefetch="BACKWARD_PRE",  # Prefetch for better performance
)

trainer.train(epochs=10)
```

> [!IMPORTANT]
> **FSDP Key Options**
>
> - **Sharding strategies**:
>     - `FULL_SHARD`: Shard everything (parameters, gradients, optimizer states) - most memory efficient
>     - `SHARD_GRAD_OP`: Shard gradients and optimizer only - faster than FULL_SHARD
>     - `HYBRID_SHARD`: PyTorch strategy for hybrid setups; OLM's documented multi-node workflow is planned for v4
>     - `NO_SHARD`: No sharding (equivalent to DDP)
> - **Auto-wrap policies**:
>     - `"size"`: Wraps layers based on parameter count (use `min_num_params` to control)
>     - `"transformer"`: Wraps specific transformer layer classes (provide `transformer_layer_cls`)
>     - `None`: Manual wrapping (you must wrap model yourself before passing to trainer)
> - **Mixed precision**: Use `"bf16"` for Ampere+ GPUs, `"fp16"` for older GPUs
> - **CPU offload**: Saves GPU memory but slows training ~2-3x

**Checkpoint Saving with FSDP**

```python
# Save full model checkpoint (only rank 0 saves)
trainer.save_checkpoint(
    path="./checkpoints/model.pt",
    state_dict_type="FULL_STATE_DICT"  # Gathers full model on rank 0
)

# Alternative: Save sharded checkpoint (all ranks save their shard)
trainer.save_checkpoint(
    path="./checkpoints/model_sharded",
    state_dict_type="SHARDED_STATE_DICT"
)
```

**Choosing Between DDP and FSDP**

| Scenario                        | Recommendation                                      |
| ------------------------------- | --------------------------------------------------- |
| Model fits on single GPU        | Use **DDP** (simpler, faster)                       |
| Model doesn't fit on single GPU | Use **FSDP** with `FULL_SHARD`                      |
| Multi-node training             | Planned for **v4**                                  |
| Maximum throughput              | Use **DDP** or **FSDP** with `SHARD_GRAD_OP`        |
| Maximum model size              | Use **FSDP** with `FULL_SHARD` + `cpu_offload=True` |

---

<h3>6. Automatic Trainer Selection (AutoTrainer)</h3>

The `AutoTrainer` provides intelligent automatic selection between single-GPU training, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) based on your hardware and model characteristics.

**Why Use AutoTrainer?**

Traditional distributed training requires you to:

- Manually detect GPU count
- Choose between DDP and FSDP
- Configure distributed backends
- Set up process groups
- Handle device placement

AutoTrainer does all of this automatically with a single parameter: `device="auto"`

**Basic Usage**

```python
from olm.train import AutoTrainer
from olm.train.optim import AdamW

# That's it! AutoTrainer handles everything
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",  # Magic!
    context_length=2048,
    learning_rate=3e-4
)

trainer.train(epochs=10)
```

**What Happens Automatically:**

1. **Hardware Detection**: Scans for available GPUs and their memory
2. **Strategy Selection**: Chooses the optimal trainer:
    - 0-1 GPU → `Trainer` (single device)
    - 2-4 GPUs, small model → `DDPTrainer`
    - 2-4 GPUs, large model → `FSDPTrainer` (HYBRID_SHARD)
    - 5+ GPUs, any model → `FSDPTrainer` (FULL_SHARD)
3. **Configuration**: Sets up distributed backend, sharding, mixed precision
4. **Initialization**: Handles `setup_distributed()` and device placement

**Configuration Presets**

Use presets to optimize for different scenarios:

```python
# Balanced (default): Smart selection based on hardware and model
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    preset="balanced",  # Default
    ...
)

# Memory Efficient: Prioritize FSDP, enable CPU offload
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    preset="memory_efficient",  # For large models
    ...
)

# Speed: Prioritize DDP, no offload, larger comm buckets
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    preset="speed",  # For maximum throughput
    ...
)
```

**Device Options**

```python
# Full auto-detection (recommended)
trainer = AutoTrainer(model=model, device="auto", ...)

# Force CUDA with auto-configuration
trainer = AutoTrainer(model=model, device="cuda:auto", ...)

# Force CPU with auto-configuration
trainer = AutoTrainer(model=model, device="cpu:auto", ...)

# Legacy mode (backward compatible)
trainer = AutoTrainer(model=model, device="cuda", ...)  # Single GPU
trainer = AutoTrainer(model=model, device="cuda:0", ...)  # Specific GPU
```

**Manual Device Detection**

For more control, you can inspect hardware before training:

```python
from olm.train import detect_devices, estimate_model_size

# Detect available hardware
config = detect_devices(verbose=True)
print(f"Found {config.num_gpus} GPUs")
print(f"GPU Memory: {config.gpu_memory_per_device:.2f} GB per device")

# Estimate model memory requirements
memory_info = estimate_model_size(model, verbose=True)
print(f"Model requires ~{memory_info['total_gb']:.2f} GB")

# Use detected config
trainer = AutoTrainer(model=model, device=config, ...)
```

**Force Specific Strategy**

Override automatic selection when needed:

```python
from olm.train import TrainerStrategy

# Force DDP even on 8 GPUs
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    ...
)

# Available strategies:
# - TrainerStrategy.SINGLE_GPU
# - TrainerStrategy.SINGLE_CPU
# - TrainerStrategy.MULTI_GPU_DDP
# - TrainerStrategy.MULTI_GPU_FSDP_HYBRID  # Within-node sharding
# - TrainerStrategy.MULTI_GPU_FSDP_FULL    # Full sharding
```

**Launching Multi-GPU Training**

AutoTrainer works seamlessly with `torchrun`:

```python
# train_script.py
from olm.train import AutoTrainer

trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",  # Automatically configures for distributed
    context_length=2048,
    ...
)
trainer.train(epochs=10)
```

Launch with torchrun:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_script.py

# Multi-node launch helpers are planned for v4.
```

**Advanced Configuration**

Fine-tune DDP or FSDP parameters:

```python
trainer = AutoTrainer(
    model=model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    context_length=2048,
    preset="memory_efficient",
    # DDP parameters (used if DDP is selected)
    ddp_find_unused_parameters=False,
    ddp_broadcast_buffers=True,
    ddp_bucket_cap_mb=25,
    # FSDP parameters (used if FSDP is selected)
    fsdp_min_num_params=100_000_000,  # 100M params for auto-wrap
    fsdp_backward_prefetch="BACKWARD_PRE",
    ...
)
```

> [!TIP]
> **When to Use Each Trainer Directly:**
>
> - Use `Trainer` for single GPU or CPU development
> - Use `DDPTrainer` when you need explicit DDP control
> - Use `FSDPTrainer` for very large models (>13B parameters)
> - Use `AutoTrainer` for everything else (recommended)

> [!NOTE]
> **Backward Compatibility:**
> AutoTrainer is fully backward compatible. Existing code using `device="cuda"` or `device="cuda:0"` continues to work without changes.

---

<h3>7. Saving and Loading</h3>

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

<h3>8. Experiment Tracking with Weights & Biases</h3>

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
    log_gradients=True,
    watch_model=True,
    watch_freq=100,
)
```

**2. Model Checkpoint Artifacts**

Automatically save and version your checkpoints:

```python
wandb_callback = WandBCallback(
    project="my-project",
    log_model=True,
)
```

**3. Alert Integration**

Get notified when metrics cross thresholds:

```python
wandb_callback = WandBCallback(
    project="my-project",
    alert_thresholds={
        "loss": {"max": 5.0},
        "learning_rate": {"min": 1e-6},
    },
)
```

**4. Prediction Table Logging**

Log model predictions for qualitative analysis:

```python
# During training, log predictions periodically
wandb_callback.log_predictions(
    step=trainer.global_step,
    inputs=["The quick brown", "Once upon a time"],
    predictions=["fox jumped over", "there was a"],
    targets=["fox jumped", "there was"],
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

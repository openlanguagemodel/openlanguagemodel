<h2> Datasets and Training </h2>

The `olm` library is designed to handle massive amounts of text data without using up all your computer's memory. It does this by "streaming" the data—reading it bit by bit as the model needs it, rather than loading everything at once. This allows you to train on datasets that are much larger than your hard drive or RAM.

---

<h3>1. Preparing Your Data</h3>

To start training, you first need to tell the library where your text is. We have three main ways to do this:

*   **From Local Files**: If you have a folder full of `.txt` files, use `LocalTextDataset`. It scans the directory and streams each file one by one.
*   **From Hugging Face**: If you want to use a dataset from the web (like Wikipedia or Common Crawl), use `HuggingFaceTextDataset`. It downloads chunks of data as you train.
*   **FineWeb Edu**: A built-in shortcut for a high-quality educational dataset, pre-configured with the best settings.

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
> You can override any of these methods: `on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`, and `on_step_begin/end`. The entire `trainer` object is passed into these, giving you access to the model, the optimizer, and the current state (like `trainer.global_step`).



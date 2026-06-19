# `olm.logging.wandb_logger`

Source: [`src/olm/logging/wandb_logger.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L1)

Weights & Biases (wandb) integration for OLM training.

Provides comprehensive logging, tracking, and monitoring capabilities using wandb.

## Functions

### `create_sweep(sweep_config: Dict[str, Any], project: str, entity: str | None = None) -> str`

Source: [`src/olm/logging/wandb_logger.py:442`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L442)

Create a wandb sweep for hyperparameter optimization.

Args:
    sweep_config: Sweep configuration dictionary.
    project: WandB project name.
    entity: WandB entity (team/username).

Returns:
    Sweep ID to use with wandb agent.

Example:
    >>> sweep_config = {
    ...     "method": "bayes",
    ...     "metric": {"name": "train/loss", "goal": "minimize"},
    ...     "parameters": {
    ...         "learning_rate": {
    ...             "distribution": "log_uniform_values",
    ...             "min": 1e-5,
    ...             "max": 1e-3,
    ...         },
    ...         "batch_size": {"values": [8, 16, 32, 64]},
    ...         "weight_decay": {
    ...             "distribution": "uniform",
    ...             "min": 0.0,
    ...             "max": 0.3,
    ...         },
    ...     },
    ... }
    >>>
    >>> sweep_id = create_sweep(sweep_config, project="my-llm-project")
    >>> print(f"Run: wandb agent {sweep_id}")

### `get_sweep_config_template(method: str = 'bayes') -> Dict[str, Any]`

Source: [`src/olm/logging/wandb_logger.py:487`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L487)

Get a template sweep configuration.

Args:
    method: Sweep method ("grid", "random", "bayes"). Default: "bayes".

Returns:
    Template sweep configuration dictionary.

Example:
    >>> config = get_sweep_config_template("bayes")
    >>> # Customize the config
    >>> config["parameters"]["learning_rate"]["min"] = 1e-5
    >>> config["parameters"]["learning_rate"]["max"] = 1e-3
    >>> # Create sweep
    >>> sweep_id = create_sweep(config, project="my-project")

## Classes

### `WandBCallback(project: str, entity: str | None = None, name: str | None = None, tags: List[str] | None = None, notes: str | None = None, config: Dict[str, Any] | None = None, log_frequency: int = 1, log_gradients: bool = False, log_model: bool = False, watch_model: bool = False, watch_freq: int = 1000, log_predictions: bool = False, log_system_metrics: bool = True, alert_thresholds: Dict[str, Dict[str, float]] | None = None, offline: bool = False, resume: str | None = None, group: str | None = None, job_type: str | None = 'train', save_code: bool = True, reinit: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/logging/wandb_logger.py:23`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L23)

Callback for Weights & Biases integration with OLM Trainer.

Provides comprehensive experiment tracking including:
- Training metrics (loss, perplexity, learning rate, throughput)
- Hyperparameter logging
- System metrics (GPU memory, CPU usage)
- Gradient and weight histograms (optional)
- Model checkpoint artifacts
- Prediction tables (optional)
- Alert monitoring (optional)
- Sweep support for hyperparameter optimization

Args:
    project: WandB project name.
    entity: WandB team/username (defaults to your default entity).
    name: Run name (auto-generated if None).
    tags: List of tags for this run.
    notes: Optional notes/description for this run.
    config: Hyperparameters and config to log (auto-captured from trainer if None).
    log_frequency: Log metrics every N steps (default: 1).
    log_gradients: Enable gradient histogram logging (can slow training).
    log_model: Save model checkpoints as wandb artifacts.
    watch_model: Use wandb.watch() for automatic gradient/parameter tracking.
    watch_freq: Frequency for wandb.watch logging (default: 1000).
    log_predictions: Enable prediction table logging.
    log_system_metrics: Log GPU/CPU metrics (default: True).
    alert_thresholds: Dict of metric thresholds for alerts.
        Example: {"loss": {"min": 0.1, "max": 10.0}}
    offline: Run in offline mode (for air-gapped environments).
    resume: Resume from previous run ("allow", "must", "never", or "auto").
    group: Group name for grouping runs.
    job_type: Job type (e.g., "train", "eval", "sweep").
    save_code: Save training code to wandb (default: True).
    reinit: Allow multiple wandb.init() calls in same process.

Example:
    >>> from olm.logging import WandBCallback
    >>>
    >>> # Basic usage
    >>> wandb_callback = WandBCallback(
    ...     project="my-llm-project",
    ...     name="llama-7b-baseline",
    ...     tags=["llama", "baseline"],
    ... )
    >>>
    >>> trainer = Trainer(..., callbacks=[wandb_callback])
    >>> trainer.train(epochs=10)
    >>>
    >>> # Advanced: with alerts and gradient logging
    >>> wandb_callback = WandBCallback(
    ...     project="my-llm-project",
    ...     log_gradients=True,
    ...     watch_model=True,
    ...     alert_thresholds={
    ...         "loss": {"max": 10.0},  # Alert if loss > 10
    ...         "learning_rate": {"min": 1e-6}  # Alert if LR < 1e-6
    ...     },
    ... )

#### Methods

##### `log_predictions(self, step: int, inputs: List[str], predictions: List[str], targets: List[str] | None = None)`

Source: [`src/olm/logging/wandb_logger.py:414`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L414)

Log predictions to wandb table.

Args:
    step: Current training step.
    inputs: Input texts.
    predictions: Model predictions.
    targets: Target texts (optional).

##### `on_epoch_end(self, trainer, epoch: int) -> None`

Source: [`src/olm/logging/wandb_logger.py:347`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L347)

Called at the end of each epoch.

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/logging/wandb_logger.py:317`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L317)

Called at the end of each optimization step.

##### `on_train_begin(self, trainer) -> None`

Source: [`src/olm/logging/wandb_logger.py:298`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L298)

Called at the beginning of training.

##### `on_train_end(self, trainer) -> None`

Source: [`src/olm/logging/wandb_logger.py:304`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/logging/wandb_logger.py#L304)

Called at the end of training.

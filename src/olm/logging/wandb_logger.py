"""
Weights & Biases (wandb) integration for OLM training.

Provides comprehensive logging, tracking, and monitoring capabilities using wandb.
"""

from typing import Optional, Dict, Any, List, Union
import os
import warnings

from olm.train.trainer.trainer import TrainerCallback


# Check if wandb is installed
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBCallback(TrainerCallback):
    """
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
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_frequency: int = 1,
        log_gradients: bool = False,
        log_model: bool = False,
        watch_model: bool = False,
        watch_freq: int = 1000,
        log_predictions: bool = False,
        log_system_metrics: bool = True,
        alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        offline: bool = False,
        resume: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = "train",
        save_code: bool = True,
        reinit: bool = True,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: "
                "pip install openlanguagemodel[wandb] or pip install wandb"
            )

        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.notes = notes
        self.config = config or {}
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        self.log_model = log_model
        self.watch_model = watch_model
        self.watch_freq = watch_freq
        self.log_predictions = log_predictions
        self.log_system_metrics = log_system_metrics
        self.alert_thresholds = alert_thresholds or {}
        self.offline = offline
        self.resume = resume
        self.group = group
        self.job_type = job_type
        self.save_code = save_code
        self.reinit = reinit

        self.run = None
        self.prediction_table = None
        self._alerts_configured = False
        self._is_distributed = False
        self._should_log = True  # Only rank 0 logs in distributed training

    def _check_distributed(self):
        """Check if we're in distributed training and if this is rank 0."""
        try:
            from olm.core.dist import is_distributed, is_main_process

            self._is_distributed = is_distributed()
            self._should_log = not self._is_distributed or is_main_process()
        except ImportError:
            self._is_distributed = False
            self._should_log = True

    def _init_wandb(self, trainer):
        """Initialize wandb run."""
        if not self._should_log:
            return  # Don't initialize on non-main processes

        # Set offline mode if requested
        if self.offline:
            os.environ["WANDB_MODE"] = "offline"

        # Build config from trainer
        config = {
            "model": {
                "context_length": trainer.context_length,
                "total_params": sum(p.numel() for p in trainer.model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in trainer.model.parameters() if p.requires_grad
                ),
            },
            "optimizer": {
                "type": type(trainer.optimizer).__name__,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            },
            "training": {
                "batch_size": trainer.dataloader.batch_size,
                "grad_accum_steps": trainer.grad_accum_steps,
                "use_amp": trainer.use_amp,
                "grad_clip_norm": trainer.grad_clip_norm,
                "warmup_steps": trainer.warmup_steps,
                "total_steps": trainer.total_steps,
                "min_lr": trainer.min_lr,
                "device": trainer.device,
            },
        }

        # Merge with user-provided config
        config.update(self.config)

        # Initialize wandb
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            notes=self.notes,
            config=config,
            resume=self.resume,
            group=self.group,
            job_type=self.job_type,
            save_code=self.save_code,
            reinit=self.reinit,
        )

        # Watch model if requested
        if self.watch_model:
            wandb.watch(
                trainer.model,
                log="all" if self.log_gradients else "gradients",
                log_freq=self.watch_freq,
                log_graph=False,  # Graph logging can be slow for large models
            )

        # Initialize prediction table if requested
        if self.log_predictions:
            self.prediction_table = wandb.Table(
                columns=["step", "input_text", "prediction", "target"]
            )

    def _configure_alerts(self):
        """Configure wandb alerts based on thresholds."""
        if not self._should_log or self._alerts_configured:
            return

        for metric_name, thresholds in self.alert_thresholds.items():
            if "min" in thresholds:
                wandb.alert(
                    title=f"{metric_name} too low",
                    text=f"{metric_name} dropped below {thresholds['min']}",
                    level=wandb.AlertLevel.WARN,
                    wait_duration=300,  # 5 minutes between alerts
                )

            if "max" in thresholds:
                wandb.alert(
                    title=f"{metric_name} too high",
                    text=f"{metric_name} exceeded {thresholds['max']}",
                    level=wandb.AlertLevel.WARN,
                    wait_duration=300,
                )

        self._alerts_configured = True

    def _check_alerts(self, metrics: Dict[str, float]):
        """Check if any metrics violate alert thresholds."""
        if not self._should_log or not self.alert_thresholds:
            return

        for metric_name, value in metrics.items():
            if metric_name not in self.alert_thresholds:
                continue

            thresholds = self.alert_thresholds[metric_name]

            if "min" in thresholds and value < thresholds["min"]:
                wandb.alert(
                    title=f"⚠️ {metric_name} too low!",
                    text=f"{metric_name} = {value:.4f} (threshold: {thresholds['min']})",
                    level=wandb.AlertLevel.ERROR,
                )

            if "max" in thresholds and value > thresholds["max"]:
                wandb.alert(
                    title=f"⚠️ {metric_name} too high!",
                    text=f"{metric_name} = {value:.4f} (threshold: {thresholds['max']})",
                    level=wandb.AlertLevel.ERROR,
                )

    def _log_system_metrics(self):
        """Log system-level metrics (GPU memory, CPU usage)."""
        if not self._should_log or not self.log_system_metrics:
            return

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    wandb.log(
                        {
                            f"system/gpu_{i}_memory_allocated_gb": torch.cuda.memory_allocated(
                                i
                            )
                            / 1e9,
                            f"system/gpu_{i}_memory_reserved_gb": torch.cuda.memory_reserved(
                                i
                            )
                            / 1e9,
                            f"system/gpu_{i}_utilization": (
                                torch.cuda.utilization(i)
                                if hasattr(torch.cuda, "utilization")
                                else 0
                            ),
                        }
                    )
        except Exception as e:
            warnings.warn(f"Failed to log system metrics: {e}")

    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        self._check_distributed()
        self._init_wandb(trainer)
        self._configure_alerts()

    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        if not self._should_log:
            return

        # Log final prediction table if enabled
        if self.log_predictions and self.prediction_table is not None:
            wandb.log({"predictions": self.prediction_table})

        # Mark run as finished
        if self.run is not None:
            wandb.finish()

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Called at the end of each optimization step."""
        if not self._should_log:
            return

        # Only log at specified frequency
        if step % self.log_frequency != 0:
            return

        # Prepare metrics
        metrics = {
            "train/loss": trainer.training_state.get("current_loss", loss),
            "train/perplexity": trainer.training_state.get("perplexity", 0),
            "train/learning_rate": trainer.training_state.get("learning_rate", 0),
            "train/tokens_per_sec": trainer.training_state.get("tokens_per_sec", 0),
            "train/epoch": trainer.current_epoch,
            "train/step": step,
            "train/total_tokens": trainer.total_tokens_processed,
        }

        # Log to wandb
        wandb.log(metrics, step=step)

        # Log system metrics periodically (every 10 log steps)
        if step % (self.log_frequency * 10) == 0:
            self._log_system_metrics()

        # Check alerts
        self._check_alerts(metrics)

    def on_epoch_end(self, trainer, epoch: int) -> None:
        """Called at the end of each epoch."""
        if not self._should_log:
            return

        # Log epoch-level summary
        wandb.log(
            {
                "epoch": epoch,
                "epoch_end_step": trainer.global_step,
            }
        )

        # Save model checkpoint if requested
        if self.log_model and epoch % 1 == 0:  # Save every epoch
            self._save_checkpoint(trainer, epoch)

    def _save_checkpoint(self, trainer, epoch: int):
        """Save model checkpoint as wandb artifact."""
        if not self._should_log:
            return

        try:
            import torch
            import tempfile
            import os

            # Create artifact
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata={
                    "epoch": epoch,
                    "step": trainer.global_step,
                    "loss": trainer.losses[-1] if trainer.losses else None,
                },
            )

            # Save checkpoint to temporary file
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": (
                            trainer.scheduler.state_dict()
                            if trainer.scheduler
                            else None
                        ),
                        "epoch": epoch,
                        "step": trainer.global_step,
                        "loss": trainer.losses[-1] if trainer.losses else None,
                    },
                    checkpoint_path,
                )

                # Add file to artifact
                artifact.add_file(checkpoint_path)

            # Log artifact
            wandb.log_artifact(artifact)

        except Exception as e:
            warnings.warn(f"Failed to save checkpoint to wandb: {e}")

    def log_predictions(
        self,
        step: int,
        inputs: List[str],
        predictions: List[str],
        targets: Optional[List[str]] = None,
    ):
        """
        Log predictions to wandb table.

        Args:
            step: Current training step.
            inputs: Input texts.
            predictions: Model predictions.
            targets: Target texts (optional).
        """
        if not self._should_log or not self.log_predictions:
            return

        if self.prediction_table is None:
            return

        # Add rows to table
        for i, (inp, pred) in enumerate(zip(inputs, predictions)):
            tgt = targets[i] if targets and i < len(targets) else ""
            self.prediction_table.add_data(step, inp, pred, tgt)


def create_sweep(
    sweep_config: Dict[str, Any],
    project: str,
    entity: Optional[str] = None,
) -> str:
    """
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
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install with: pip install wandb")

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    return sweep_id


def get_sweep_config_template(method: str = "bayes") -> Dict[str, Any]:
    """
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
    """
    return {
        "method": method,
        "metric": {
            "name": "train/loss",
            "goal": "minimize",
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "batch_size": {
                "values": [8, 16, 32, 64],
            },
            "weight_decay": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.3,
            },
            "warmup_steps": {
                "values": [100, 500, 1000, 2000],
            },
            "grad_accum_steps": {
                "values": [1, 2, 4, 8],
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
        },
    }

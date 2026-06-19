# `olm.train.callbacks.throughput_cb`

Source: [`src/olm/train/callbacks/throughput_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L1)

Throughput monitoring callback.

## Classes

### `ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/throughput_cb.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L9)

Callback to monitor training throughput (tokens/sec, samples/sec).

Args:
    log_every: Log throughput every N steps.
    context_length: Length of each sequence.
    batch_size: Total batch size (including gradient accumulation).

#### Methods

##### `on_step_begin(self, trainer, step: int) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L30)

Record start time of the step.

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L34)

Calculate and log throughput.

# `olm.train.optim.zero`

Source: [`src/olm/train/optim/zero.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L1)

## Classes

### `ZeROOptimizer(optimizer: torch.optim.optimizer.Optimizer, partition_optimizer_states: bool = True, overlap_communication: bool = True, world_size: int | None = None, rank: int | None = None)`

**Bases:** `olm.train.optim.base.OptimizerBase`

Source: [`src/olm/train/optim/zero.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L9)

ZeRO (Zero Redundancy Optimizer) wrapper for distributed training.

Implements memory optimization techniques from "ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020).

ZeRO reduces memory consumption by partitioning optimizer states, gradients,
and parameters across data-parallel processes. This implementation provides
a simplified version focusing on optimizer state partitioning (ZeRO Stage 1).

For full ZeRO support with gradient and parameter partitioning, consider using
DeepSpeed or PyTorch's FSDP (Fully Sharded Data Parallel).

**Parameters**

- `optimizer`: Base optimizer to wrap (e.g., AdamW, Lion)
- `partition_optimizer_states`: Whether to partition optimizer states (default: True)
- `overlap_communication`: Overlap gradient communication with computation (default: True)
- `world_size`: Number of distributed processes (default: None, auto-detected)
- `rank`: Process rank in distributed group (default: None, auto-detected)

#### Properties

- `defaults`
  Access underlying optimizer's defaults.
- `param_groups`
  Access underlying optimizer's parameter groups.
- `state`
  Access underlying optimizer's state.

#### Methods

##### `add_param_group(self, param_group: Dict[str, Any])`

Source: [`src/olm/train/optim/zero.py:204`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L204)

Add a param group to the Optimizer's param_groups.

**Parameters**

- `param_group`: parameter group to add

##### `extra_repr(self) -> str`

Source: [`src/olm/train/optim/zero.py:227`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L227)

String representation for debugging.

##### `load_state_dict(self, state_dict: Dict[str, Any])`

Source: [`src/olm/train/optim/zero.py:137`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L137)

Loads the optimizer state.

**Parameters**

- `state_dict`: optimizer state dict

##### `state_dict(self) -> Dict[str, Any]`

Source: [`src/olm/train/optim/zero.py:107`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L107)

Returns the state of the optimizer as a dict.

In distributed mode, only returns states for parameters owned by this rank.

##### `zero_grad(self, set_to_none: bool = True)`

Source: [`src/olm/train/optim/zero.py:194`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/zero.py#L194)

Sets gradients of all optimized parameters to zero.

**Parameters**

- `set_to_none`: instead of setting to zero, set the grads to None.
- `Default`: True (overriding base class to match modern PyTorch conventions)

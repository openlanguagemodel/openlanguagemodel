# API Reference

Generated from the public Python API in `src/olm`.
Each module page includes signatures, docstrings, and source-defined methods such as `forward()` where available.

## Core

| Module | Public API |
|---|---|
| [`olm.core.dist`](generated/olm.core.dist.md) | `all_gather`, `all_reduce`, `barrier`, `broadcast`, `cleanup_distributed`, `get_backend`, `get_local_rank`, `get_rank`, +6 more |
| [`olm.core.registry`](generated/olm.core.registry.md) | `Registry` |

## Data

| Module | Public API |
|---|---|
| [`olm.data.datasets`](generated/olm.data.datasets.md) | `BaseTextDataset`, `DataLoader`, `FineWebEduDataset`, `HuggingFaceTextDataset`, `LocalTextDataset` |
| [`olm.data.datasets.base_dataset`](generated/olm.data.datasets.base_dataset.md) | `BaseTextDataset` |
| [`olm.data.datasets.data_loader`](generated/olm.data.datasets.data_loader.md) | `DataLoader` |
| [`olm.data.datasets.fineweb_edu`](generated/olm.data.datasets.fineweb_edu.md) | `FineWebEduDataset` |
| [`olm.data.datasets.hf_dataset`](generated/olm.data.datasets.hf_dataset.md) | `FineWebEduDataset`, `HuggingFaceTextDataset` |
| [`olm.data.datasets.local_dataset`](generated/olm.data.datasets.local_dataset.md) | `LocalTextDataset` |
| [`olm.data.tokenization.base`](generated/olm.data.tokenization.base.md) | `TokenizerBase` |
| [`olm.data.tokenization.hf_tokenizer`](generated/olm.data.tokenization.hf_tokenizer.md) | `HFTokenizer` |
| [`olm.data.tokenization.hf_train_custom`](generated/olm.data.tokenization.hf_train_custom.md) | `HFTokenizerTrainCustom` |

## Logging

| Module | Public API |
|---|---|
| [`olm.logging`](generated/olm.logging.md) | `WandBCallback`, `create_sweep`, `get_sweep_config_template` |
| [`olm.logging.wandb_logger`](generated/olm.logging.wandb_logger.md) | `WandBCallback`, `create_sweep`, `get_sweep_config_template` |

## Models

| Module | Public API |
|---|---|
| [`olm.models`](generated/olm.models.md) | `GPT2`, `GPT2Large`, `GPT2Medium`, `GPT2Model`, `GPT2XL`, `Gemma2Model`, `Gemma2_27B`, `Gemma2_2B`, +28 more |
| [`olm.models.alibaba`](generated/olm.models.alibaba.md) | `Qwen2Model`, `Qwen2_5_0_5B`, `Qwen2_5_14B`, `Qwen2_5_1_5B`, `Qwen2_5_32B`, `Qwen2_5_3B`, `Qwen2_5_72B`, `Qwen2_5_7B` |
| [`olm.models.alibaba.qwen2`](generated/olm.models.alibaba.qwen2.md) | `Qwen2Block`, `Qwen2Model`, `Qwen2_5_0_5B`, `Qwen2_5_14B`, `Qwen2_5_1_5B`, `Qwen2_5_32B`, `Qwen2_5_3B`, `Qwen2_5_72B`, +1 more |
| [`olm.models.allenai`](generated/olm.models.allenai.md) | `OLMoModel`, `OLMo_7B` |
| [`olm.models.allenai.olmo`](generated/olm.models.allenai.olmo.md) | `OLMoBlock`, `OLMoModel`, `OLMo_7B` |
| [`olm.models.facebook`](generated/olm.models.facebook.md) | `OPT125M`, `OPTModel` |
| [`olm.models.facebook.opt`](generated/olm.models.facebook.opt.md) | `OPT125M`, `OPTBlock`, `OPTModel` |
| [`olm.models.google`](generated/olm.models.google.md) | `Gemma2Model`, `Gemma2_27B`, `Gemma2_2B`, `Gemma2_9B` |
| [`olm.models.google.gemma2`](generated/olm.models.google.gemma2.md) | `Gemma2Block`, `Gemma2Embedding`, `Gemma2FinalLogitSoftcap`, `Gemma2Model`, `Gemma2_27B`, `Gemma2_2B`, `Gemma2_9B` |
| [`olm.models.meta`](generated/olm.models.meta.md) | `Llama2Model`, `Llama2_13B`, `Llama2_70B`, `Llama2_7B`, `Llama3Model`, `Llama3_1_405B`, `Llama3_1_70B`, `Llama3_1_8B`, +2 more |
| [`olm.models.meta.llama2`](generated/olm.models.meta.llama2.md) | `Llama2Block`, `Llama2Model`, `Llama2_13B`, `Llama2_70B`, `Llama2_7B` |
| [`olm.models.meta.llama3`](generated/olm.models.meta.llama3.md) | `Llama3Block`, `Llama3Model`, `Llama3_1_405B`, `Llama3_1_70B`, `Llama3_1_8B`, `Llama3_2_1B`, `Llama3_2_3B` |
| [`olm.models.microsoft`](generated/olm.models.microsoft.md) | `Phi3Model`, `Phi3_5_Mini`, `Phi3_Small`, `Phi4Model`, `Phi4_14B` |
| [`olm.models.microsoft.phi3`](generated/olm.models.microsoft.phi3.md) | `Phi3Block`, `Phi3Model`, `Phi3_5_Mini`, `Phi3_Small` |
| [`olm.models.microsoft.phi4`](generated/olm.models.microsoft.phi4.md) | `Phi4Block`, `Phi4Model`, `Phi4_14B` |
| [`olm.models.openai`](generated/olm.models.openai.md) | `GPT2`, `GPT2Large`, `GPT2Medium`, `GPT2Model`, `GPT2XL` |
| [`olm.models.openai.gpt2`](generated/olm.models.openai.gpt2.md) | `GPT2`, `GPT2Block`, `GPT2Large`, `GPT2Medium`, `GPT2Model`, `GPT2XL` |

## Neural Network Components

| Module | Public API |
|---|---|
| [`olm.nn.activations.base`](generated/olm.nn.activations.base.md) | `ActivationBase` |
| [`olm.nn.activations.elu`](generated/olm.nn.activations.elu.md) | `ELU` |
| [`olm.nn.activations.geglu`](generated/olm.nn.activations.geglu.md) | `GeGLU` |
| [`olm.nn.activations.gelu`](generated/olm.nn.activations.gelu.md) | `GELU` |
| [`olm.nn.activations.glu`](generated/olm.nn.activations.glu.md) | `GLU` |
| [`olm.nn.activations.identity`](generated/olm.nn.activations.identity.md) | `Identity` |
| [`olm.nn.activations.leaky_relu`](generated/olm.nn.activations.leaky_relu.md) | `LeakyReLU` |
| [`olm.nn.activations.liglu`](generated/olm.nn.activations.liglu.md) | `LiGLU` |
| [`olm.nn.activations.mish`](generated/olm.nn.activations.mish.md) | `Mish` |
| [`olm.nn.activations.prelu`](generated/olm.nn.activations.prelu.md) | `PReLU` |
| [`olm.nn.activations.reglu`](generated/olm.nn.activations.reglu.md) | `ReGLU` |
| [`olm.nn.activations.relu`](generated/olm.nn.activations.relu.md) | `ReLU` |
| [`olm.nn.activations.selu`](generated/olm.nn.activations.selu.md) | `SELU` |
| [`olm.nn.activations.sigmoid`](generated/olm.nn.activations.sigmoid.md) | `Sigmoid` |
| [`olm.nn.activations.silu`](generated/olm.nn.activations.silu.md) | `SiLU`, `Swish` |
| [`olm.nn.activations.softmax`](generated/olm.nn.activations.softmax.md) | `Softmax` |
| [`olm.nn.activations.softplus`](generated/olm.nn.activations.softplus.md) | `Softplus` |
| [`olm.nn.activations.swiglu`](generated/olm.nn.activations.swiglu.md) | `SwiGLU` |
| [`olm.nn.activations.swish`](generated/olm.nn.activations.swish.md) | `Swish` |
| [`olm.nn.activations.tanh`](generated/olm.nn.activations.tanh.md) | `Tanh` |
| [`olm.nn.attention`](generated/olm.nn.attention.md) | `AttentionBase`, `AttentionwithRoPEBase`, `FlashAttention`, `FlashAttentionwithRoPE`, `GroupedQueryAttention`, `MultiHeadAttention`, `MultiHeadAttentionwithALiBi`, `MultiHeadAttentionwithRoPE` |
| [`olm.nn.attention.alibi`](generated/olm.nn.attention.alibi.md) | `MultiHeadAttentionwithALiBi` |
| [`olm.nn.attention.base`](generated/olm.nn.attention.base.md) | `AttentionBase`, `AttentionwithRoPEBase` |
| [`olm.nn.attention.flash`](generated/olm.nn.attention.flash.md) | `FlashAttention`, `FlashAttentionwithRoPE` |
| [`olm.nn.attention.gqa`](generated/olm.nn.attention.gqa.md) | `GroupedQueryAttention` |
| [`olm.nn.attention.masks`](generated/olm.nn.attention.masks.md) | `attention_mask_to_bool` |
| [`olm.nn.attention.mha`](generated/olm.nn.attention.mha.md) | `MultiHeadAttention`, `MultiHeadAttentionwithRoPE` |
| [`olm.nn.blocks.LM`](generated/olm.nn.blocks.LM.md) | `LM` |
| [`olm.nn.blocks.linear_projections`](generated/olm.nn.blocks.linear_projections.md) | `QKVProjection` |
| [`olm.nn.blocks.output_head`](generated/olm.nn.blocks.output_head.md) | `OutputHead` |
| [`olm.nn.blocks.transformer_block`](generated/olm.nn.blocks.transformer_block.md) | `TransformerBlock` |
| [`olm.nn.embeddings`](generated/olm.nn.embeddings.md) | `AbsolutePositionalEmbedding`, `Embedding` |
| [`olm.nn.embeddings.positional`](generated/olm.nn.embeddings.positional.md) | `ALiBiPositionalBias`, `AbsolutePositionalEmbedding`, `PartialRotaryPositionalEmbedding`, `PositionalEmbeddingBase`, `RotaryPositionalEmbedding`, `SinusoidalPositionalEmbedding` |
| [`olm.nn.embeddings.positional.absolute`](generated/olm.nn.embeddings.positional.absolute.md) | `AbsolutePositionalEmbedding` |
| [`olm.nn.embeddings.positional.alibi`](generated/olm.nn.embeddings.positional.alibi.md) | `ALiBiPositionalBias` |
| [`olm.nn.embeddings.positional.base`](generated/olm.nn.embeddings.positional.base.md) | `PositionalEmbeddingBase` |
| [`olm.nn.embeddings.positional.rope`](generated/olm.nn.embeddings.positional.rope.md) | `PartialRotaryPositionalEmbedding`, `PartialScaledRotaryPositionalEmbedding`, `RotaryPositionalEmbedding`, `ScaledRotaryPositionalEmbedding` |
| [`olm.nn.embeddings.positional.sinusoidal`](generated/olm.nn.embeddings.positional.sinusoidal.md) | `SinusoidalPositionalEmbedding` |
| [`olm.nn.embeddings.token_embed`](generated/olm.nn.embeddings.token_embed.md) | `Embedding` |
| [`olm.nn.feedforward`](generated/olm.nn.feedforward.md) | `ClassicFFN`, `ClassicMoEFFN`, `FeedForwardBase`, `GeGLUFFN`, `GeGLUMoEFFN`, `SwiGLUFFN`, `SwiGLUMoEFFN` |
| [`olm.nn.feedforward.base`](generated/olm.nn.feedforward.base.md) | `FeedForwardBase` |
| [`olm.nn.feedforward.classic_ffn`](generated/olm.nn.feedforward.classic_ffn.md) | `ClassicFFN` |
| [`olm.nn.feedforward.classic_moe`](generated/olm.nn.feedforward.classic_moe.md) | `ClassicMoEFFN` |
| [`olm.nn.feedforward.geglu_ffn`](generated/olm.nn.feedforward.geglu_ffn.md) | `GeGLUFFN` |
| [`olm.nn.feedforward.geglu_moe`](generated/olm.nn.feedforward.geglu_moe.md) | `GeGLUMoEFFN` |
| [`olm.nn.feedforward.moe_base`](generated/olm.nn.feedforward.moe_base.md) | `MoEFeedForwardBase`, `MoERouter` |
| [`olm.nn.feedforward.swiglu_ffn`](generated/olm.nn.feedforward.swiglu_ffn.md) | `SwiGLUFFN` |
| [`olm.nn.feedforward.swiglu_moe`](generated/olm.nn.feedforward.swiglu_moe.md) | `SwiGLUMoEFFN` |
| [`olm.nn.norms`](generated/olm.nn.norms.md) | `LayerNorm`, `RMSNorm` |
| [`olm.nn.norms.base`](generated/olm.nn.norms.base.md) | `NormBase` |
| [`olm.nn.norms.layer_norm`](generated/olm.nn.norms.layer_norm.md) | `LayerNorm` |
| [`olm.nn.norms.rms_norm`](generated/olm.nn.norms.rms_norm.md) | `RMSNorm` |
| [`olm.nn.structure.block`](generated/olm.nn.structure.block.md) | `Block`, `load`, `load_block`, `load_model` |
| [`olm.nn.structure.combinators`](generated/olm.nn.structure.combinators.md) | `BaseCombinator`, `Parallel`, `Repeat`, `Residual` |
| [`olm.nn.structure.combinators.base`](generated/olm.nn.structure.combinators.base.md) | `BaseCombinator` |
| [`olm.nn.structure.combinators.parallel`](generated/olm.nn.structure.combinators.parallel.md) | `Parallel` |
| [`olm.nn.structure.combinators.repeat`](generated/olm.nn.structure.combinators.repeat.md) | `Repeat` |
| [`olm.nn.structure.combinators.residual`](generated/olm.nn.structure.combinators.residual.md) | `Residual` |
| [`olm.nn.torch_nn_wrappers`](generated/olm.nn.torch_nn_wrappers.md) | `Linear` |

## Training

| Module | Public API |
|---|---|
| [`olm.train`](generated/olm.train.md) | `AdamW`, `CheckpointCallback`, `CosineAnnealingLR`, `DDPTrainer`, `DeviceConfig`, `EarlyStoppingCallback`, `FSDPTrainer`, `LRMonitorCallback`, +20 more |
| [`olm.train.callbacks`](generated/olm.train.callbacks.md) | `CheckpointCallback`, `EarlyStoppingCallback`, `LRMonitorCallback`, `MetricsLoggerCallback`, `ThroughputCallback`, `ValidationCallback` |
| [`olm.train.callbacks.checkpoint_cb`](generated/olm.train.callbacks.checkpoint_cb.md) | `CheckpointCallback` |
| [`olm.train.callbacks.early_stopping_cb`](generated/olm.train.callbacks.early_stopping_cb.md) | `EarlyStoppingCallback` |
| [`olm.train.callbacks.lr_monitor_cb`](generated/olm.train.callbacks.lr_monitor_cb.md) | `LRMonitorCallback` |
| [`olm.train.callbacks.metrics_logger_cb`](generated/olm.train.callbacks.metrics_logger_cb.md) | `MetricsLoggerCallback` |
| [`olm.train.callbacks.throughput_cb`](generated/olm.train.callbacks.throughput_cb.md) | `ThroughputCallback` |
| [`olm.train.callbacks.validation_cb`](generated/olm.train.callbacks.validation_cb.md) | `ValidationCallback` |
| [`olm.train.device`](generated/olm.train.device.md) | `DeviceConfig`, `TrainerStrategy`, `detect_devices`, `determine_strategy`, `estimate_model_size`, `parse_device_string`, `print_strategy_summary` |
| [`olm.train.optim`](generated/olm.train.optim.md) | `AdamW`, `Lion`, `OptimizerBase`, `ZeROOptimizer` |
| [`olm.train.optim.adamw`](generated/olm.train.optim.adamw.md) | `AdamW` |
| [`olm.train.optim.base`](generated/olm.train.optim.base.md) | `OptimizerBase` |
| [`olm.train.optim.lion`](generated/olm.train.optim.lion.md) | `Lion` |
| [`olm.train.optim.zero`](generated/olm.train.optim.zero.md) | `ZeROOptimizer` |
| [`olm.train.schedulers`](generated/olm.train.schedulers.md) | `CosineAnnealingLR`, `LinearDecayLR`, `LinearLR`, `SchedulerBase`, `WarmupCosineScheduler`, `WarmupLR` |
| [`olm.train.schedulers.base`](generated/olm.train.schedulers.base.md) | `SchedulerBase` |
| [`olm.train.schedulers.cosine`](generated/olm.train.schedulers.cosine.md) | `CosineAnnealingLR` |
| [`olm.train.schedulers.linear`](generated/olm.train.schedulers.linear.md) | `LinearDecayLR`, `LinearLR` |
| [`olm.train.schedulers.warmup`](generated/olm.train.schedulers.warmup.md) | `WarmupCosineScheduler`, `WarmupLR` |
| [`olm.train.trainer`](generated/olm.train.trainer.md) | `CheckpointCallback`, `DDPTrainer`, `EarlyStoppingCallback`, `FSDPTrainer`, `LRMonitorCallback`, `MetricsLoggerCallback`, `ThroughputCallback`, `Trainer`, +4 more |
| [`olm.train.trainer.auto_trainer`](generated/olm.train.trainer.auto_trainer.md) | `AutoTrainer`, `auto_trainer` |
| [`olm.train.trainer.ddp_trainer`](generated/olm.train.trainer.ddp_trainer.md) | `DDPTrainer` |
| [`olm.train.trainer.fsdp_trainer`](generated/olm.train.trainer.fsdp_trainer.md) | `FSDPTrainer` |
| [`olm.train.trainer.trainer`](generated/olm.train.trainer.trainer.md) | `Trainer`, `TrainerCallback` |

# Version-wise Roadmap

This document outlines the development trajectory of OpenLanguageModel (OLM). Our goal is to move from a flexible single-GPU research tool to a scalable, high-performance distributed training framework.

## v1.0: Foundation & Core Architectures

The focus of v1.0 is to establish a solid, bug-free foundation for single-GPU training and to support a diverse set of standard architecture families.

- [ ] **Core Architecture Support**:
    - [ ] GPT-2 (Base, Medium, Large, XL)
    - [ ] OLMo (standard and variant architectures)
    - [ ] Phi-3 / Phi-4 (including variable grouped-query attention)
    - [ ] Gemma 2 (incorporating specific normalization and gating nuances)
- [ ] **Data Pipeline**:
    - [ ] Unified `LocalTextDataset` and `HuggingFaceTextDataset` interfaces
    - [ ] Efficient streaming and tokenization
    - [ ] Robust train/validation splitting
- [ ] **Training Engine**:
    - [ ] Mixed Precision Training (AMP) support
    - [ ] Gradient Clipping and Weight Decay integration
    - [ ] Basic learning rate scheduling (Cosine, Linear)
    - [ ] Checkpoint saving and loading
- [ ] **Infrastructure**:
    - [ ] Comprehensive Unit Tests for all core modules
    - [ ] CI/CD pipeline setup

## v1.1: Optimization & Refinement

v1.1 targets extracting maximum performance from a single GPU and enhancing the user experience.

- [ ] **Performance Optimization**:
    - [ ] Flash Attention integration
    - [ ] `torch.compile` compatibility
    - [ ] Memory optimization (activation checkpointing)
- [ ] **Advanced Architectures**:
    - [ ] Support for RoPE scaling variations
    - [ ] ALiBi positional embeddings
    - [ ] Custom activation wrappers refinement
- [ ] **Observability**:
    - [ ] Integration with Weights & Biases (WandB)
    - [ ] Detailed training logs and metrics (perplexity, tokens/sec)
- [ ] **Documentation**:
    - [ ] Full API reference
    - [ ] End-to-end tutorials and notebooks

## v2.0: Scaling Up (Multi-GPU)

v2.0 introduces distributed training capabilities, allowing OLM to utilize multiple GPUs on a single node.

- [ ] **Distributed Training**:
    - [x] Distributed Data Parallel (DDP) support
    - [x] Fully Sharded Data Parallel (FSDP) integration
- [ ] **Mixture of Experts (MoE)**:
    - [ ] Sparse MoE layer implementation
    - [ ] Top-k gating mechanisms
    - [ ] Load balancing auxiliary losses

## v2.1: Distributed Optimization

v2.1 focuses on making multi-GPU training highly efficient and stable.

- [ ] **Efficiency**:
    - [ ] Zero Redundancy Optimizer (ZeRO) stages
    - [ ] Efficient communication overlap
- [ ] **MoE Enhancements**:
    - [ ] Expert parallelism
    - [ ] Expert routing optimization

## v3.0: Scaling Out (Multi-Node)

v3.0 expands the horizon to multi-node clusters for training large-scale models.

- [ ] **Cluster Support**:
    - [ ] Slurm integration
    - [ ] Fault tolerance and auto-resume
    - [ ] Multi-node data streaming handling
- [ ] **Large Model Training**:
    - [ ] Pipeline Parallelism
    - [ ] Tensor Parallelism

## v3.1: The "Open Source" Goal

v3.1 aims to make OLM capable of reproducing widely used open-source models from scratch.

- [ ] **Reproduction Recipes**:
    - [ ] Verified configs to retrain Llama 3, Mistral, etc. from scratch
- [ ] **Ecosystem**:
    - [ ] Evaluation harness integration (HellaSwag, MMLU)
    - [ ] Quantization-aware training
    
## v4.0: Further Training

- [ ] Implement SFT
- [ ] Implement LoRA
- [ ] Implement RLHF, DDP & RLVR methods
- [ ] Implement model fine-tuning

## Other Ideas
- [ ] Visual model builder: Drag and drop components to build models
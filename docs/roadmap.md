# Version-wise Roadmap

This document outlines the development trajectory of OpenLanguageModel (OLM). Our goal is to move from a flexible single-GPU research tool to a scalable, high-performance distributed training framework.

## v1.0: Foundation & Core Architectures

The focus of v1.0 is to establish a solid, bug-free foundation for single-GPU training and to support a diverse set of standard architecture families.

- [x] **Core Architecture Support**:
    - [x] GPT-2 (Base, Medium, Large, XL)
    - [x] OLMo (standard and variant architectures)
    - [x] Phi-3 / Phi-4 (including variable grouped-query attention)
    - [x] Gemma 2 (incorporating specific normalization and gating nuances)
- [x] **Data Pipeline**:
    - [x] Unified `LocalTextDataset` and `HuggingFaceTextDataset` interfaces
    - [x] Efficient streaming and tokenization
    - [x] Robust train/validation splitting
- [x] **Training Engine**:
    - [x] Mixed Precision Training (AMP) support
    - [x] Gradient Clipping and Weight Decay integration
    - [x] Basic learning rate scheduling (Cosine, Linear)
    - [x] Checkpoint saving and loading
- [x] **Infrastructure**:
    - [x] Comprehensive Unit Tests for all core modules
    - [x] CI/CD pipeline setup

## v1.1: Optimization & Refinement

v1.1 targets extracting maximum performance from a single GPU and enhancing the user experience.

- [x] **Performance Optimization**:
    - [x] Flash Attention integration
    - [x] `torch.compile` compatibility
    - [x] Memory optimization (activation checkpointing)
- [x] **Advanced Architectures**:
    - [x] Support for RoPE scaling variations
    - [x] ALiBi positional embeddings
    - [x] Custom activation wrappers refinement
- [x] **Observability**:
    - [x] Integration with Weights & Biases (WandB)
    - [x] Detailed training logs and metrics (perplexity, tokens/sec)
- [ ] **Documentation**:
    - [ ] Full API reference
    - [ ] End-to-end tutorials and notebooks

## v2.0: Scaling Up (Multi-GPU)

v2.0 introduces distributed training capabilities, allowing OLM to utilize multiple GPUs on a single node.

- [x] **Distributed Training**:
    - [x] Distributed Data Parallel (DDP) support
    - [x] Fully Sharded Data Parallel (FSDP) integration
- [x] **Mixture of Experts (MoE)**:
    - [x] Sparse MoE layer implementation
    - [x] Top-k gating mechanisms
    - [x] Load balancing auxiliary losses

## v2.1: Distributed Optimization

v2.1 focuses on making multi-GPU training highly efficient and stable.

- [x] **Efficiency**:
    - [x] Zero Redundancy Optimizer (ZeRO) stages
    - [x] Efficient communication overlap
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
# Roadmap

OpenLanguageModel is moving toward a stable, readable, PyTorch-native stack for language-model learning, ablation, and training. The near-term goal is not to add more surface area; it is to make the existing library feel clean, dependable, and easy to teach from.

## v2.2: Stability, Documentation, and Release Readiness

**Current release.** v2.2 is the stabilization release. The focus is a polished library and documentation set, not new research features.

- [x] Move onto the v2.1 bug-fix and AutoTrainer base
- [x] Stabilize single-node multi-GPU training with DDP/FSDP paths
- [x] Make model output heads tied to token embeddings by default
- [x] Verify model-family smoke tests and one-batch training paths
- [x] Improve README positioning, citation, model links, and install guidance
- [x] Regenerate and improve API reference structure
- [x] Remove empty placeholders and stale public references
- [x] Add clear examples for local text, FineWeb-Edu, AutoTrainer, and model families
- [x] Track website source and GitHub Pages deployment workflow
- [x] Add website SEO metadata, sitemap, robots, social preview, and structured data
- [ ] Finish mascot integration after the mascot direction is chosen
- [x] Add approved Colab notebooks and link them from OLM Learning
- [x] Prepare the public v2.2 release notes and PyPI/GitHub release checklist

## v3.0: Further Training and Alignment

v3 is for post-pretraining workflows. The goal is to let people continue from a pretrained or base model while staying inside ordinary PyTorch.

- [ ] Supervised fine-tuning (SFT) recipes and trainers
- [ ] LoRA and parameter-efficient fine-tuning
- [ ] Preference optimization with DPO
- [ ] RLHF workflows with PPO
- [ ] RLVR / reasoning-oriented training with GRPO-style methods
- [ ] Evaluation hooks for common language-model and instruction-following tasks
- [ ] Checkpoint conversion and compatibility guidance for fine-tuned models

## v4.0: Multi-Node Training

v4 moves beyond v2's single-node multi-GPU support into cluster-scale training. The intent is to keep the user-facing API understandable while exposing the distributed systems pieces needed for serious runs.

- [ ] Multi-node launch and configuration helpers
- [ ] Slurm and common cluster integration
- [ ] Fault-tolerant checkpointing and auto-resume
- [ ] Multi-node streaming and deterministic data sharding
- [ ] Multi-node FSDP recipes and performance guidance

## Longer-Term Ideas

These are intentionally outside v2.2.

- Verified reproduction recipes for open-source model families
- More model-family implementations, including Mistral-style and DeepSeek-style variants
- A visual model builder for composing blocks interactively
- Export and conversion tooling once the core training/documentation path is fully stable

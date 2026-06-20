# OpenLanguageModel v2.2.0

v2.2.0 is a stabilization release for OpenLanguageModel. The focus is reliability, documentation, packaging, and a cleaner path from first install to real language-model training.

## Highlights

- Tied token embeddings are now the default for OLM language models and model presets.
- `AutoTrainer` is available for hardware-aware trainer selection across CPU, GPU, DDP, and FSDP-capable setups.
- Single-node multi-GPU training paths are documented and covered by focused tests.
- Streaming text datasets avoid special-token insertion during continuous language-model training.
- Final partial gradient-accumulation windows are now flushed instead of being silently dropped.
- FSDP checkpoint saving has clearer full, local, and sharded checkpoint behavior.
- Model-family presets have broader smoke coverage, including one-batch trainability checks.
- Llama, Qwen, Phi, Gemma, OLMo, OPT, and GPT-2 implementations are linked from the website and docs.
- The generated API reference now includes public losses, tokenizer behavior, embeddings, trainers, and source-linked forward methods.
- The website, docs, README, sitemap, metadata, and Colab notebook index have been updated for the v2.2 release.

## Install

```bash
pip install openlanguagemodel==2.2.0
```

For source installs:

```bash
git clone https://github.com/openlanguagemodel/openlanguagemodel.git
cd openlanguagemodel
pip install -e .
```

OLM v2.2 supports Python 3.10, 3.11, and 3.12.

## Verify

```bash
python - <<'PY'
import olm
from olm.nn.blocks import LM

print("olm", olm.__version__)
model = LM(vocab_size=128, embed_dim=32, num_heads=4, num_layers=1, max_seq_len=16)
print(type(model).__name__)
PY
```

## Notes

v2.2 is intentionally not a feature-expansion release. Multi-node training and cluster support remain on the v4 roadmap. Alignment and post-pretraining workflows such as SFT, LoRA, DPO, PPO/RLHF, and GRPO-style RLVR are planned for v3.

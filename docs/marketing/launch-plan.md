# Launch And Publicity Plan

OLM's public message should be simple:

> OpenLanguageModel is a PyTorch-native LLM library for learning, ablations,
> and language-model training.

## Launch Goals

- Turn the v2.2 release into GitHub stars, PyPI installs, and course interest.
- Make the notebooks and learning path easy to find.
- Give researchers a reason to inspect the architecture code.
- Give instructors a reason to try OLM for a language-modeling module.

## Primary Launch Post

Draft title:

> OLM: PyTorch-native language model training for learning and ablations

Core points:

- model architecture stays visible as ordinary PyTorch modules
- training stack includes datasets, tokenizers, AMP, checkpointing, AutoTrainer,
  and single-node DDP/FSDP paths
- model families include GPT-2, Llama, Qwen, Phi, Gemma, OLMo, and OPT
- the learning path includes docs, Colabs, and a from-scratch course
- v2.2 is focused on stability, docs, website, and release polish

Links to include:

- Website: `https://openlanguagemodel.github.io/openlanguagemodel/`
- GitHub: `https://github.com/openlanguagemodel/openlanguagemodel`
- PyPI: `https://pypi.org/project/openlanguagemodel/`
- Colabs: `https://openlanguagemodel.github.io/openlanguagemodel/docs/colab-notebooks/`

## Channels

- Hacker News: Show HN style post
- Reddit: `r/MachineLearning`, `r/LocalLLaMA`, `r/learnmachinelearning`, and
  course/education communities where allowed
- LinkedIn: educator/researcher framing
- X: concise demo clips or code snippets
- ML Discords and Slack communities
- University course outreach from maintainers

## Evergreen Articles

The first five articles should target useful non-brand searches:

- Build a GPT-2 Style Language Model from Scratch in PyTorch
- How Transformer Blocks Work: Attention, MLPs, Norms, and Residuals
- How to Train a 125M Language Model on FineWeb-Edu
- Custom Transformer Architectures in PyTorch Without Rewriting the Training Loop
- DDP vs FSDP for Language Model Training

Each article should link to docs, Colabs, source files, and a runnable example.

## University Outreach

For instructors, lead with:

- a walkable language-modeling course
- Colab notebooks that avoid setup pain
- readable PyTorch-native architecture code
- a path from tiny local models to real FineWeb-Edu training

Ask for:

- feedback on the course
- links from course pages or lab notes if they adopt it
- GitHub stars from students who use it
- issues when anything is confusing

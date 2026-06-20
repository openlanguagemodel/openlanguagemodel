# Learn Language Modelling From Scratch

A guided path from a little Python and a little deep learning to building and
training your own language model — and understanding every piece. You write and
run real code from the very first lesson, using [OLM](../getting-started.md).

> [!NOTE]
> **Already comfortable with the basics and just want to train a model?** Skip
> ahead to [Your First Language Model](../tutorials/first-model.md) and
> [Custom Architectures](../tutorials/custom-architecture.md) in **Start Building**.

## Who this is for

This course is for you if you can write a bit of Python and have seen a neural
network at least once — enough to know roughly what a tensor and a *weight* are.
If you're a second-year CS student who has taken one introductory ML course,
you're exactly who it's written for. Everything specific to language models —
tokens, embeddings, attention, transformers, how training works — we build up from
the beginning, one lesson at a time.

## Where you'll get to

- Explain how a language model works and how it is trained.
- Train a small GPT-style model on your own text and generate from it.
- Read a modern architecture (Llama, Qwen) and recognise every component.
- Assemble your own architecture from OLM's building blocks.
- Know where to go next — scaling up, multiple GPUs, research.

## The path

Work through the lessons in order; each is short and builds on the last. Keep
[Key Concepts](../concepts.md) handy for a quick definition of any term.

### Foundations

- **[Lesson 0 · Set Up Your Lab](setup.md)** — somewhere to run code (Google Colab,
  nothing to install) and your first line of OLM.
- **[Lesson 1 · What Is a Language Model?](what-is-a-language-model.md)** — the one
  idea the whole field is built on, and how text becomes numbers.
- **[Lesson 2 · Words as Vectors](words-as-vectors.md)** — embeddings, and the
  surprising thing: meaning becomes geometry. You'll *see* it.
- **[Lesson 3 · Paying Attention](paying-attention.md)** — how words reshape each
  other's meaning, and what attention is for.
- **[Lesson 4 · A Whole Transformer Block](a-transformer-block.md)** — attention,
  feed-forward, and normalization together: the unit every LLM stacks.
- **[Lesson 5 · How a Model Learns](how-a-model-learns.md)** — what training really
  is: guess, measure the error, nudge, repeat.
- **[Deep dive · How Attention Works](attention-in-detail.md)** *(optional)* — open
  the box: queries, keys, values, and the rest.

### Build for real

Once you've got the foundations, these tutorials take you the rest of the way:

- **Train your first model** → [Your First Language Model](../tutorials/first-model.md)
- **Design your own architecture** → [The Block System](../guides/architecture.md)
  and [Custom Architectures](../tutorials/custom-architecture.md)
- **Scale up** → [Datasets & Training](../guides/datasets-and-training.md),
  [Distributed Training](../tutorials/distributed-training.md),
  [Experiment Tracking](../tutorials/experiment-tracking.md)
- **Capstone** → pretrain a GPT-2 from scratch; see the full project in
  [`examples/gpt2-fineweb-edu-10b`](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/examples/gpt2-fineweb-edu-10b)

## How this course is different

Most resources teach language modelling starting from the mathematics. This one
starts from building. From the first lessons you assemble and train real models
with OLM's components, and pick up the theory alongside — as deep as you want to go.
Curious what's under a step? Each lesson has optional *going deeper* notes. Just want
to build? Skip them and you'll still come away with working models. It's the approach
that made PyTorch approachable for deep learning: useful first, fundamental when
you're ready.

## For educators & institutions

OLM is built to be *taught*. The library is small and readable, every component maps
one-to-one to a concept, and this course already sequences intuition → code →
practice. That makes it a ready-made backbone for a semester course or lab module, a
workshop or bootcamp unit, or self-study cohorts and reading groups.

Because the same Markdown powers this site and the
[GitHub repository](https://github.com/openlanguagemodel/openlanguagemodel), students
can read along wherever they work, and the code is the textbook. If you're considering
OLM for a course and want help adapting this path — or want to contribute lessons,
exercises, or solutions — [open an issue or reach out on GitHub](https://github.com/openlanguagemodel/openlanguagemodel).

> [!NOTE]
> The foundation lessons (0–5) and the attention deep dive are all available now.
> Exercises and further material are planned.

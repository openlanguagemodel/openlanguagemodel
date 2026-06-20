# Key Concepts

This page explains the words you will meet while training a language model with
OLM — in plain English, one idea at a time. You do **not** need to read it all
before training your [first model](tutorials/first-model.md). Keep it open in a
tab and come back whenever a term is unfamiliar.

Each section ends with a pointer to where OLM lets you actually use the idea, so
you can go as deep as you want.

> [!TIP]
> Everything here is standard across modern language models. Once these click,
> you will understand not just OLM but papers and other libraries too. That is
> the point — OLM is meant to teach the skill, not hide it.

## What a language model actually does

A language model does one small thing, over and over: **given some text so far,
predict the next word.** "The cat sat on the …" → *"mat"*.

That is it. Everything else — writing essays, answering questions, generating
code — is this single prediction repeated. You feed the model some text, it
predicts the next piece, you stick that piece on the end, and feed it back in.
This is called **autoregressive** generation ("auto" = self, "regressive" =
feeding its own output back).

**Training** the model means showing it billions of examples of real text and
nudging it, each time, to make its prediction of the next word a little closer
to what actually came next. This one task is called **next-token prediction**.

## Tokens and the tokenizer

Models do not see words or letters — they see **tokens**. A token is a chunk of
text, often a whole word but sometimes a piece of one (`"running"` might become
`"run"` + `"ning"`). The component that chops text into tokens is the
**tokenizer**.

Every token has an integer ID, like a row number in a dictionary. So the
tokenizer turns `"the cat"` into a list of numbers like `[464, 3797]`. The model
only ever works with these numbers.

- **Vocabulary size** (`vocab_size`) — how many distinct tokens exist, i.e. how
  big that dictionary is. GPT-2's is about 50,000.

In OLM you create one with [`HFTokenizer`](api/data.md), e.g.
`HFTokenizer("gpt2")`, and ask it for `tokenizer.vocab_size`.

## Embeddings

A token ID like `3797` is just a label; it carries no meaning by itself. An
**embedding** turns each token ID into a list of numbers (a vector) that the
model can do math on. Similar tokens end up with similar vectors — the model
learns this during training.

- **Embedding dimension** (`embed_dim`) — how long that list of numbers is, e.g.
  768. Bigger means the model can represent more nuance, but costs more compute.
  It is the model's "width," and it stays the same all the way through.

See [Building Blocks → Embeddings](guides/components.md) for the variants OLM
ships.

## Context length and `max_seq_len`

- **Context length** — how many tokens the model reads at once. A context length
  of 128 means it sees the last 128 tokens when predicting the next one. Longer
  context lets the model "remember" more, but costs more memory and compute.
- **`max_seq_len`** — the **largest** context length the model was *built* to
  support. Some components (positional information, below) need to know this
  maximum up front to size their internal tables.

They are related but not the same: `max_seq_len` is the ceiling you build in;
context length is how much you actually feed in a given run. Keep
`max_seq_len` **greater than or equal to** the context length you train with.

## Attention (and "heads")

**Attention** is the mechanism that lets the model look back at earlier tokens
and decide which ones matter for predicting the next one. When completing
"The cat sat on the …", attention lets the model focus on "cat" and "sat" rather
than every word equally. It is the core idea behind transformers.

- **Attention heads** (`num_heads`) — attention is run several times in parallel,
  each "head" free to focus on a different kind of relationship (one might track
  grammar, another long-range topic). Their results are combined. More heads =
  more kinds of relationships tracked at once.
- **Causal** — for next-token prediction the model must only look *backward*, never
  at future tokens (that would be cheating). "Causal" attention enforces this.

OLM has a whole family — basic multi-head attention, grouped-query attention
(GQA), and more — compared side by side in
[Building Blocks → Attention](guides/components.md).

## Positional information: RoPE, ALiBi, and friends

Attention on its own does not know the **order** of tokens — "dog bites man" and
"man bites dog" would look the same. So models add **positional information**.

You will see these names; here is the one-line version of each:

- **Absolute / sinusoidal** — add a position signal to each token's embedding.
  The classic, simple approach.
- **RoPE** (Rotary Positional Embeddings) — encodes position by *rotating* the
  vectors inside attention. It is what most modern models (Llama, Qwen) use
  because it generalizes well to longer sequences.
- **ALiBi** — gently penalizes attention to far-away tokens instead of adding a
  position signal.

You do not need to understand the math to use them — pick one when you build a
model. See [Building Blocks → Positional embeddings](guides/components.md).

## Feed-forward, and "SwiGLU"

After attention gathers information, each token is passed through a small neural
network — the **feed-forward** layer (FFN) — that transforms it further. Roughly:
attention mixes information *between* tokens; the feed-forward layer processes
each token on its own.

- **SwiGLU / GeGLU** — these are just particular, slightly fancier feed-forward
  designs that tend to train better. "SwiGLU" = a feed-forward layer using a
  gated activation (the gate decides how much of each signal to let through).
  Modern LLMs use these instead of the plain version.

The plain and gated variants are in
[Building Blocks → Feed-forward](guides/components.md).

## Normalization: LayerNorm and RMSNorm

Deep networks train more stably if the numbers flowing through them are kept at a
sensible scale. **Normalization** layers do this rescaling at each step.

- **LayerNorm** — the standard approach.
- **RMSNorm** — a simpler, faster variant used by most recent models (Llama,
  Qwen). Same purpose.

You will see one of these between the attention and feed-forward parts of every
block. More in [Building Blocks → Normalization](guides/components.md).

## Logits

When the model finishes processing, it outputs one score for every token in the
vocabulary — how strongly it believes that token comes next. These raw scores
are called **logits**.

Turn logits into probabilities (with a softmax) and you get "12% chance it's
*mat*, 4% chance it's *floor*, …". To actually pick a next token you **sample**
from those probabilities — see [generating text](tutorials/first-model.md).

## Loss and perplexity

During training you need a number that says "how wrong was that prediction?" —
that number is the **loss**. Lower is better. Training works by repeatedly
nudging the model to make the loss go down.

- **Perplexity** — a friendlier way to read the loss. Roughly, "how surprised was
  the model by the correct next token?" A perplexity of 20 means the model was
  about as unsure as if it were guessing among 20 equally likely words.
  Perplexity of 1 would be perfect. Like loss, **lower is better** — watching it
  fall is how you know training is working.

OLM's [`Trainer`](api/train.md) prints both as it trains.

## How training runs: the moving parts

A few terms describe *how* the training loop is run. OLM's `Trainer` handles all
of these for you, but here is what they mean:

- **Epoch** — one full pass over your training data. `max_steps` instead caps
  training at a fixed number of update steps.
- **Learning rate** — how big a step the model takes each time it adjusts itself.
  Too big and training is unstable; too small and it crawls.
- **Warmup + cosine schedule** — rather than a fixed learning rate, it is common
  to start small, ramp **up** for a few hundred steps ("warmup"), then smoothly
  decay it along a cosine curve. This is just a recipe for changing the learning
  rate over time, and it trains more stably.
- **Mixed precision (AMP)** — doing some of the math in a lower-precision number
  format so it runs faster and uses less GPU memory, with almost no quality loss.
  "AMP" = Automatic Mixed Precision.
- **Gradient accumulation** — if your GPU cannot fit a big batch, process several
  small batches and add up their adjustments before taking one step. It
  simulates a large batch on small hardware.
- **Gradient clipping** — capping how large a single adjustment can be, so one bad
  batch cannot blow up training.

All of these are options on the [`Trainer`](api/train.md); the
[Datasets & Training guide](guides/datasets-and-training.md) shows them in use.

## OLM components are just PyTorch

One thing worth knowing early: **every OLM building block is a plain
`torch.nn.Module`.** An attention layer, a feed-forward layer, a whole model —
they are ordinary PyTorch objects.

That means you are never locked in. You can drop an OLM component into your own
PyTorch code, train it with your own loop, mix it with other libraries, or use
OLM's [`Trainer`](api/train.md) for convenience. OLM is a library you *call*, not
a framework that takes over. As you grow from beginner to researcher, nothing you
learn here is wasted.

## Where to go next

- **Just want to train something?** → [Your First Language Model](tutorials/first-model.md)
- **Want to see every component?** → [Building Blocks](guides/components.md)
- **Want to design your own architecture?** → [The Block System](guides/architecture.md)
  and [Custom Architectures](tutorials/custom-architecture.md)
- **Want exact signatures and options?** → [API Reference](api/index.md)

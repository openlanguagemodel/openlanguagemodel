# Lesson 0 · Set Up Your Lab

Before we build anything, you need a place to run code. This lesson gets you one
in about five minutes — no prior setup, and nothing to install on your computer.

By the end you'll have run your first line of OLM and seen a sentence turn into
numbers.

> [!NOTE]
> This whole course assumes **zero** setup experience. If a step seems obvious,
> skip it. If a step is new, you're exactly who we wrote it for.

## The easiest option: Google Colab

**Google Colab** is a free tool that lets you write and run Python *in your web
browser*. Your code actually runs on Google's computers, so:

- there's **nothing to install**,
- it works the same on any laptop (Windows, Mac, Chromebook),
- and you can get a **free GPU** later, when we start training (a GPU is a chip
  that makes training much faster — more on that when we need it).

It's the standard way people learn and prototype machine learning. All you need
is a Google account.

### Open your first notebook

1. Go to **[colab.research.google.com](https://colab.research.google.com)**.
2. Click **New notebook** (you may need to sign in to your Google account).

You now have an empty **notebook** — a page made of **cells**. A cell is just a
box you type code into. You run a cell by clicking it and pressing
**Shift + Enter**. The result appears right underneath.

### Try it

Click the first cell, type this, and press **Shift + Enter**:

```python
print("Hello, language models!")
2 + 2
```

You should see the text printed, and `4` underneath. That's it — you're running
Python. Everything in this course works exactly like this: read a little, run a
cell, see what happens.

## Install OLM

In a new cell, run:

```python
!pip install openlanguagemodel
```

A couple of notes for newcomers:

- The `!` at the start tells Colab "run this as a setup command, not as Python."
  You only use it for installing things.
- This downloads OLM and the libraries it needs. It takes a minute or two the
  first time. You'll see a lot of text scroll by — that's normal. When it stops,
  it's done.

> [!TIP]
> Colab forgets installed packages when you close it for a while. If you come back
> tomorrow and `import olm` fails, just run the `!pip install` cell again.

## Your first real line of OLM

Run this in a new cell:

```python
from olm.data.tokenization import HFTokenizer

tok = HFTokenizer("gpt2")
print(tok.encode("Language models are just predicting the next word."))
```

You'll see something like a list of numbers wrapped in `tensor([...])`. You just
turned a sentence into numbers — the form a model can actually work with. **We'll
unpack exactly what those numbers are in Lesson 1.** For now, the point is simply:
it ran, and you saw output.

## (Optional) Turn on the free GPU

You won't need it until we start training, but here's how, so you know where it
is:

1. In the Colab menu: **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, pick a **GPU** (e.g. "T4 GPU").
3. Click **Save**.

That's it. We'll come back to this in the training lessons.

## Prefer to work on your own machine?

That's completely fine — Colab is just the gentlest start. If you already have
Python set up locally and would rather work there, follow the
[Getting Started](../getting-started.md#installation) guide instead, then come
back here. Everything in the course runs identically either way.

## What you learned

- **Colab** lets you run Python in the browser with nothing to install.
- You run code one **cell** at a time with **Shift + Enter**.
- `!pip install openlanguagemodel` adds OLM to your notebook.
- A sentence can be turned into numbers with a **tokenizer** — which is exactly
  where Lesson 1 begins.

**Next:** [Lesson 1 · What Is a Language Model?](what-is-a-language-model.md)

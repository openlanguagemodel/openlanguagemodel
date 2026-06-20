# Lesson 1 · What Is a Language Model?

In this lesson you'll learn the single idea the entire field is built on — and
you'll watch a sentence become numbers with your own hands.

> Make sure you finished [Lesson 0](setup.md) and can run a cell. Keep that same
> Colab notebook open; we'll keep building in it.

## One idea: guess the next word

Here's a game. Fill in the blank:

> The cat sat on the ___

You probably thought *"mat"* — or maybe *"floor"*, *"sofa"*, *"roof"*. You almost
certainly did **not** think *"democracy"* or *"seventeen"*. Without any effort,
you ranked every possible next word by how well it fits.

**That is exactly what a language model does.** Given some text so far, it
predicts what comes next. That's the whole job:

> **A language model takes a piece of text and predicts the next piece.**

Everything else you've heard about — chatbots, code assistants, story writers —
is this one trick, repeated.

## Why one trick is enough

Say the model predicts the next word, you add it to the sentence, and then you ask
it again — now including the word it just produced:

```text
"The cat sat on the"        → mat
"The cat sat on the mat"    → and
"The cat sat on the mat and" → fell
...
```

Keep going and you get whole sentences, paragraphs, essays. Because the model
keeps feeding on its own output, this is called **autoregressive** generation
(don't worry about the word — it just means "predicts the next piece, then reads
its own prediction back in"). Generating text is nothing more than running the
next-word guess over and over.

## A detail: models predict *tokens*, not words

There's one wrinkle. Models don't quite work with words — they work with
**tokens**. A token is a small chunk of text. Often a token is a whole word, but
longer or rarer words get split into pieces. For example a model might see
`"tokenization"` as `token` + `ization`.

Why bother? Splitting into reusable pieces keeps the vocabulary (the list of all
tokens the model knows) a manageable size, while still being able to spell out any
word — even ones it has never seen.

The tool that does this splitting is the **tokenizer**. Let's watch it work.

## Try it: turn text into tokens

In your notebook, run:

```python
from olm.data.tokenization import HFTokenizer

tok = HFTokenizer("gpt2")

ids = tok.encode("The cat sat on the mat.")
print(ids)
```

You'll see a `tensor([...])` of integers. Each integer is one token's **ID** — its
row number in the tokenizer's vocabulary. The sentence is now a sequence of
numbers, which is the only thing a neural network can actually take in.

Now turn the numbers back into text to prove nothing was lost:

```python
print(tok.decode(ids))
```

You get your sentence back. **`encode` goes text → numbers; `decode` goes numbers
→ text.** That round trip is the bridge between human language and the model.

### See the pieces

Try a long, unusual word and a common one, and compare how many tokens each
becomes:

```python
print(tok.encode("cat").shape)            # a common word
print(tok.encode("antidisestablish").shape)   # a rare one
```

The common word is a single token; the rare word gets chopped into several. That's
sub-word tokenization in action — frequent things get their own token, rare things
are spelled out from pieces.

> [!TIP]
> `.shape` tells you the size of a tensor. Here it's just the number of tokens.
> You'll use `.shape` constantly — it's how you check that data is the size you
> expect.

## What does it mean for a model to be *good*?

A language model doesn't output a single next token — it outputs a **score for
every token in its vocabulary**, saying how likely each one is to come next. Turn
those scores into percentages and a good model, given *"The cat sat on the"*,
might say:

| Next token | Model's confidence |
|------------|--------------------|
| ` mat`     | 31% |
| ` floor`   | 12% |
| ` sofa`    | 9%  |
| ` ground`  | 7%  |
| … 50,000 more … | … |
| ` democracy` | 0.0001% |

A **good** model puts high confidence on tokens that genuinely fit. A bad model
spreads its confidence randomly. **Training** — which we'll get to — is the
process of turning the second kind of model into the first, by showing it mountains
of real text and correcting it every time it's wrong.

Right now, a freshly made model is the bad kind: it has seen nothing, so its
guesses are basically random. Everything from here is about giving it the
machinery and the practice to get good.

## What you learned

- A language model does one thing: **predict the next token** from the text so far.
- Repeating that prediction is how all text is **generated** (autoregression).
- Text becomes **tokens** (numbers) via a **tokenizer**; `encode` and `decode` are
  the two directions.
- A model outputs a **confidence for every possible next token**, and training is
  what makes those confidences good.

But how can a list of numbers possibly capture that *"cat"* and *"dog"* are
similar, while *"cat"* and *"democracy"* are not? That's the surprisingly
beautiful idea behind **embeddings** — and it's next.

**Next:** [Lesson 2 · Words as Vectors](words-as-vectors.md)

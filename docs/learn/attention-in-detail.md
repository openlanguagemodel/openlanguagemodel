# How Attention Works (deep dive)

This is an **optional** deep dive. The course never depends on it — in
[Lesson 3](paying-attention.md) you learned everything you need to build with
attention: it lets each token gather context from the others. Here we open the box
and see exactly *how* it decides which tokens to gather from.

It's a good read once the rest of the course has settled. Take it slowly.

## The setup

Coming into an attention layer, every token is a vector (its embedding, possibly
already shaped by earlier blocks). Attention's job is to produce, for each token, a
new vector that mixes in information from the other tokens — weighting the relevant
ones heavily and the irrelevant ones lightly.

The question is: *relevant according to what?* Attention learns the answer using
three roles for every token — a **query**, a **key**, and a **value**.

## Queries, keys, and values

From each token's vector, the layer produces three new vectors, each by a small
learned linear layer:

- **Query (Q)** — what this token is *looking for*.
- **Key (K)** — what this token *offers* to others.
- **Value (V)** — the information this token will *hand over* if it's attended to.

A search analogy: your **query** is what you type in; every item carries a **key**
(a label describing it); and each item's **value** is its actual content. You match
your query against all the keys, then collect the values of whatever matched.

So for a sentence of six tokens, the layer has six queries, six keys, and six values.

## Scoring: which tokens matter

To find how relevant token *j* is to token *i*, attention compares token *i*'s query
with token *j*'s key using a **dot product** — a single number that's large when the
two vectors point in similar directions. Do this for every pair and you get, for each
token, a score against every other token.

Those raw scores are divided by the square root of the head's dimension (a fixed
number) to keep them from growing too large as vectors get longer — this keeps the
next step well-behaved. Then a **softmax** turns each token's row of scores into
**weights** between 0 and 1 that add up to 1. Big score → big weight.

These weights are the "attention" — for each token, how much it will draw from each
other token.

## Blending: a weighted sum of values

Finally, each token's output is the **weighted sum of all the value vectors**, using
its attention weights. A token that scored "river" highly pulls in mostly river's
value; the tokens it scored low barely contribute. The result is a new vector for
each token, blended from the ones that matter to it.

That's the entire mechanism. In one line of notation, with queries Q, keys K, and
values V:

```text
attention(Q, K, V) = softmax( (Q · Kᵀ) / √d ) · V
```

Read left to right it's just the three steps above: score (Q·Kᵀ), scale and normalize
(softmax over √d), then blend (× V).

## Looking back only: causal masking

For next-token prediction, a token must not peek at tokens that come after it. This is
enforced right before the softmax: every score from a token to a *later* token is set
to negative infinity, so after softmax its weight is exactly 0. That's all `causal=True`
does — it blocks the forward-looking connections.

## Many heads at once

One set of Q/K/V projections can only learn one notion of "relevant." So attention
runs several in parallel — each a **head** with its own learned Q/K/V layers. One head
might track grammatical agreement, another the sentence's subject, another long-range
topic. Each head does the full score-scale-blend on its own slice of the numbers, the
heads' outputs are concatenated, and a final linear layer mixes them back together.

If a model has embedding dimension 16 and 4 heads, each head works with 16 ÷ 4 = 4
numbers per token. More heads means more kinds of relationship tracked at once.

## How OLM implements it

This is exactly what OLM's attention components do. Creating one:

```python
from olm.nn.attention import MultiHeadAttention

attn = MultiHeadAttention(embed_dim=16, num_heads=4, causal=True)
```

Internally it holds the learned query, key, value, and output projections; on each
forward pass it produces Q, K, and V, runs the scaled dot-product attention above
(with the causal mask, since `causal=True`), recombines the heads, and applies the
output projection. The variants you'll see elsewhere are the same machine with one
change each:

- **`MultiHeadAttentionwithRoPE`** folds in *positional information* (word order) by
  rotating the queries and keys — what modern models like Llama and Qwen use.
- **`GroupedQueryAttention`** lets several query heads share one set of keys and
  values, which saves memory when generating text.
- **`FlashAttention`** computes the very same result with a faster, more
  memory-efficient algorithm.

You can compare them all in [Building Blocks → Attention](../guides/components.md), and
see the exact signatures in the [API reference](../api/nn.md).

## Further reading

For a longer, beautifully illustrated walk through this same machinery,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) is a
deservedly popular explainer and pairs well with everything above.

## What you learned

- Attention gives every token a **query**, **key**, and **value**.
- It **scores** each token's query against all keys, **scales and softmaxes** the
  scores into weights, then outputs a **weighted sum of values**.
- **Causal masking** zeroes out attention to future tokens for next-token prediction.
- **Multiple heads** run this in parallel to capture different relationships.
- OLM's `MultiHeadAttention` (and its RoPE, grouped-query, and Flash variants)
  implement exactly this.

**Back to:** [Lesson 5 · How a Model Learns](how-a-model-learns.md) ·
[Course Overview](index.md)

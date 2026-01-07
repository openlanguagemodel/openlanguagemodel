"""
Inference script for Generic 125M model (GPT-2–like).

Features:
- Nucleus (top-p) sampling
- Repetition penalty (generated tokens only)
- No-repeat n-gram blocking
- Stable decoding for FineWeb-Edu–trained GPT-2

Usage:
    python inference.py --checkpoint checkpoints/checkpoint_step_19000.pt
"""

import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from olm.nn.blocks import LM
from olm.data.tokenization.hf_tokenizer import HFTokenizer


# -------------------------
# Utility: no-repeat n-gram
# -------------------------
def ban_repeated_ngrams(input_ids, logits, n):
    """
    Hard block repeated n-grams (batch size = 1).
    """
    if input_ids.size(1) < n:
        return logits

    tokens = input_ids[0].tolist()
    ngrams = set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    prefix = tuple(tokens[-(n - 1):])

    for ng in ngrams:
        if ng[:-1] == prefix:
            logits[0, ng[-1]] = -float("inf")

    return logits


# -------------------------
# Model loading
# -------------------------
def load_model(
    checkpoint_path: str,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    max_seq_len: int,
    dropout: float,
    ff_multiplier: float,
    device: str,
):
    print(f"Loading model from {checkpoint_path}")

    model = LM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        ff_multiplier=ff_multiplier,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Backward compatibility
    if any(k.startswith("stack.") for k in state_dict):
        print("Converting legacy checkpoint format")
        new_sd = {}
        for k, v in state_dict.items():
            k = k.replace("stack.", "", 1)
            k = k.replace(".layers.blocks.", ".blocks.")
            new_sd[k] = v
        state_dict = new_sd

    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"Model loaded (step={checkpoint.get('step', 'unknown')})")
    return model


# -------------------------
# Text generation
# -------------------------
@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    device: str,
    max_seq_len: int,
):
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    prompt_len = input_ids.shape[1]

    print(f"\nPROMPT:\n{prompt}\n")
    print(f"Generating {max_new_tokens} tokens...\n")

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        # Repetition penalty (generated tokens only)
        if repetition_penalty != 1.0 and input_ids.shape[1] > prompt_len:
            generated_tokens = set(input_ids[0, prompt_len:].tolist())
            for tok in generated_tokens:
                if next_logits[0, tok] < 0:
                    next_logits[0, tok] *= repetition_penalty
                else:
                    next_logits[0, tok] /= repetition_penalty

        # Temperature
        next_logits /= temperature

        # No-repeat n-gram blocking
        if no_repeat_ngram_size > 0:
            next_logits = ban_repeated_ngrams(
                input_ids, next_logits, no_repeat_ngram_size
            )

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_logits[cutoff] = -float("inf")
        next_logits.scatter_(1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if input_ids.shape[1] >= max_seq_len:
            print("Reached maximum sequence length")
            break

    return tokenizer.decode(input_ids[0])


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.4)
    parser.add_argument("--no_repeat_ngram", type=int, default=3)

    # Architecture
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_multiplier", type=float, default=4.0)

    args = parser.parse_args()

    tokenizer = HFTokenizer("gpt2")

    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        ff_multiplier=args.ff_multiplier,
        device=args.device,
    )

    prompt = (
        "Newton's Second Law of Motion states that the acceleration of an object"
    )

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        device=args.device,
        max_seq_len=args.max_seq_len,
    )

    print("=" * 80)
    print("GENERATED TEXT")
    print("=" * 80)
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()

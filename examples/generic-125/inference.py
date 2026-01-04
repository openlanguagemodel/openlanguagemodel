"""
Inference script for GPT-2 model.

This script loads a trained GPT-2 model from a checkpoint and performs text generation.

Usage:
    python inference.py --checkpoint checkpoints/step_19000.pt --prompt "Once upon a time"
"""

import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from olm.models.gpt import GPT2
from olm.data.tokenization.hf_tokenizer import HFTokenizer


def load_model(checkpoint_path: str, device: str = "cuda") -> GPT2:
    """
    Load a trained GPT-2 model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on ('cuda' or 'cpu').

    Returns:
        Loaded GPT-2 model in evaluation mode.
    """
    print(f"Loading model from {checkpoint_path}...")

    # Initialize model
    model = GPT2()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully! (Step: {checkpoint.get('step', 'unknown')})")
    return model


def generate_text(
    model: GPT2,
    tokenizer: HFTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> str:
    """
    Generate text using the model.

    Args:
        model: The GPT-2 model.
        tokenizer: The tokenizer.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Keep only top k tokens with highest probability.
        top_p: Nucleus sampling - keep top tokens with cumulative probability >= top_p.
        repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition).
        device: Device to run inference on.

    Returns:
        Generated text including the prompt.
    """
    model.eval()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...\n")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

            # Get logits for the last token
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    # If score < 0, multiply by penalty (makes it more negative)
                    # If score > 0, divide by penalty (makes it less positive)
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check if we've hit max sequence length
            if input_ids.shape[1] >= 1024:  # GPT-2's max context length
                print("Warning: Reached maximum sequence length (1024 tokens)")
                break

    # Decode and return the generated text
    generated_text = tokenizer.decode(input_ids[0])
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained GPT-2 model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/step_19000.pt",
        help="Path to model checkpoint",
    )
    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     default="What is the capital of France?",
    #     help="Text prompt for generation"
    # )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (> 1.0 discourages repetition, 1.0 = no penalty)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="HuggingFace tokenizer to use"
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = HFTokenizer(args.tokenizer)

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    prompt = """Average life span in the wild: 12 years Size: 21 in (50 cm) Weight: 14.4 oz (408 g) Did you know? Chameleons don't change colors to match their surroundings. Each species displays distinct color patterns to indicate specific reactions or emotions. The Meller's chameleon is the largest of the chameleons not native to Madagascar. Their stout bodies can grow to be up to two feet (two-thirds of a meter) long and weigh more than a pound (one-half kilogram). Meller's distinguish themselves from their universally bizarre-looking cousins with a single small horn protruding from the front of their snouts. This and their size earn them the common name "giant one-horned chameleon." They are fairly common in the savanna of East Africa, including Malawi, northern Mozambique, and Tanzania. Almost one-half of the world’s chameleons live on the island of Madagascar. As with all chameleons, Meller's will change colors in response to stress and to communicate with other chameleons. Their normal appearance is deep green with yellow stripes and random black spots. Females are slightly smaller, but are otherwise indistinguishable from males. They subsist on insects and small birds, using their camouflage and a lightning-fast, catapulting tongue, which can be up to 20 inches (50 centimeters) long, to ambush prey. Exotic pet enthusiasts often attempt to keep Meller's chameleons as pets. However, they are highly susceptible to even the slightest level of stress and are very difficult to care for in captivity. In the wild, they can live as long as 12 years.
    Question. What is the average life span of a Meller's chameleon in the wild?
    Answer. 12 years
    Question. Why are Meller's chameleons difficult to keep as pets?
    Answer. They are highly susceptible to stress and difficult to care for in captivity.
    Question. What is the primary diet of Meller's chameleons?
    Answer.
"""

    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )

    print("=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)


if __name__ == "__main__":
    main()

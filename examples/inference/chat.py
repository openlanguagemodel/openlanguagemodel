
import os
import sys
import torch
import torch.nn.functional as F
from olm.nn.structure import load_model

def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_k=40, device="cuda", stop_token_id=None):
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt).to(device).unsqueeze(0)
    
    # Try to find max_seq_len if it exists in the model
    # Most transformer models have it hidden somewhere or in configs
    # We'll use a conservative default if not found
    max_context_len = 2048 
    
    for _ in range(max_new_tokens):
        # Crop input if it's too long
        if input_ids.size(1) > max_context_len:
            curr_input = input_ids[:, -max_context_len:]
        else:
            curr_input = input_ids
            
        # Forward pass
        with torch.no_grad():
            logits = model(curr_input)
            
        # Get last token logits and scale by temperature
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_id], dim=1)
        
        # Decode and yield
        token_text = tokenizer.decode(next_id)
        yield token_text
        
        # Stop if stop_token_id is reached
        if stop_token_id is not None and next_id.item() == stop_token_id:
            break
            
        # Also check for tokenizer's default EOS token
        if hasattr(tokenizer, "eos_token_id") and next_id.item() == tokenizer.eos_token_id:
            break
        # Common end of text token for GPT-2
        if next_id.item() == 50256: 
            break

def main():
    if len(sys.argv) < 2:
        print("Usage: py chat.py <locationofmodel> [stop_token_id]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    stop_token_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[INFO] Loading model from: {model_path}")
    if stop_token_id is not None:
        print(f"[INFO] Using custom stop token ID: {stop_token_id}")
    
    try:
        loaded = load_model(model_path)
        if isinstance(loaded, tuple):
            model, tokenizer = loaded
        else:
            model = loaded
            # It's possible the tokenizer wasn't saved, but for chat we NEED it.
            print("[ERROR] Tokenizer not found in model directory.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
        
    model.to(device)
    print(f"[INFO] Model loaded successfully on {device}!")
    print("-" * 50)
    print(" Welcome to OLM Chat! (Type 'exit' to quit)")
    print("-" * 50)
    
    # We keep a simple history to maintain context
    # Note: For base models, this format works but they might need more steerage
    history = []
    
    while True:
        try:
            print("\033[94mUser:\033[0m ", end="")
            user_input = input()
            
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            history.append(f"User: {user_input}")
            # Construct the prompt from history
            prompt = "\n".join(history) + "\nAssistant: "
            
            print("\033[92mAssistant:\033[0m ", end="", flush=True)
            
            assistant_response = ""
            for token_text in generate(model, tokenizer, prompt, device=device, stop_token_id=stop_token_id):
                print(token_text, end="", flush=True)
                assistant_response += token_text
            
            print("\n")
            history.append(f"Assistant: {assistant_response.strip()}")
            
            # Prevent history from growing too large
            if len(history) > 10:
                history = history[-10:]
                
        except KeyboardInterrupt:
            print("\n[INFO] Conversation ended.")
            break

if __name__ == "__main__":
    main()

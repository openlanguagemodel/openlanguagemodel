
import torch
import torch.nn as nn
import os
import shutil
from olm.nn.structure import Block, load_block
from olm.data.tokenization.hf_tokenizer import HFTokenizer

def test_tokenizer_save_load():
    print("Testing HFTokenizer save/load...")
    save_path = "test_tokenizer_save"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    # Use a small tokenizer for testing
    tokenizer = HFTokenizer("gpt2")
    text = "Hello, how are you?"
    encoded = tokenizer.encode(text)
    
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    
    # Test loading
    try:
        loaded_tokenizer = HFTokenizer.load(save_path)
        decoded = loaded_tokenizer.decode(encoded)
        
        print(f"Original text: {text}")
        print(f"Decoded text: {decoded}")
        assert text == decoded
        print("HFTokenizer save/load test passed!")
    except Exception as e:
        print(f"HFTokenizer save/load test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_block_save_load():
    print("\nTesting Block save/load...")
    save_path = "test_block_save"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    # Create a simple block
    sub_blocks = [nn.Linear(10, 10), nn.ReLU()]
    block = Block(sub_blocks)
    
    # Create some dummy data and get output
    x = torch.randn(1, 10)
    original_output = block(x)
    
    # Save the block
    try:
        block.save(save_path)
        print(f"Block saved to {save_path}")
        
        # Load the block
        # NOTE: This is expected to fail currently because Load lacks the architecture info
        loaded_block = load_block(save_path)
        loaded_output = loaded_block(x)
        
        assert torch.allclose(original_output, loaded_output)
        print("Block save/load test passed!")
    except Exception as e:
        print(f"Block save/load test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_block_with_tokenizer_save_load():
    print("\nTesting Block with Tokenizer save/load...")
    save_path = "test_block_tokenizer_save"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    tokenizer = HFTokenizer("gpt2")
    sub_blocks = [nn.Linear(10, 10)]
    block = Block(sub_blocks)
    
    try:
        block.save(save_path, tokenizer=tokenizer)
        print(f"Block and Tokenizer saved to {save_path}")
        
        loaded_block, loaded_tokenizer = load_block(save_path)
        print("Successfully loaded both block and tokenizer")
        
        # Basic check
        assert isinstance(loaded_block, Block)
        assert isinstance(loaded_tokenizer, HFTokenizer)
        print("Block with Tokenizer save/load test passed!")
    except Exception as e:
        print(f"Block with Tokenizer save/load test FAILED: {e}")
        # traceback already printed in other tests if needed

if __name__ == "__main__":
    # Ensure we are in the right directory or use absolute paths if needed
    # For now, just run it.
    test_tokenizer_save_load()
    test_block_save_load()
    test_block_with_tokenizer_save_load()

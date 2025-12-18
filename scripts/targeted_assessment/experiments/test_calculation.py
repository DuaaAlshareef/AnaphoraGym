"""
Test script to verify log-likelihood calculation correctness.

This script tests the calculate_llh function with a simple example to ensure
it's computing probabilities correctly.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_model_and_tokenizer


def calculate_llh(model, tokenizer, input_text, continuation_text):
    """
    Calculate the average log-likelihood of a continuation given an input.
    """
    if not isinstance(input_text, str) or not isinstance(continuation_text, str):
        return float('nan')
    
    device = model.device
    full_text = input_text + continuation_text
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    input_only_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    print(f"\nInput: '{input_text}'")
    print(f"Continuation: '{continuation_text}'")
    print(f"Full text: '{full_text}'")
    print(f"Input tokens: {input_only_ids.shape[1]}")
    print(f"Full tokens: {input_ids.shape[1]}")
    print(f"Continuation should be {input_ids.shape[1] - input_only_ids.shape[1]} tokens")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    continuation_start_index = input_only_ids.shape[1]
    print(f"Continuation starts at token index: {continuation_start_index}")
    
    # Get logits for continuation tokens
    # logits[i] predicts token[i+1], so we need logits[continuation_start_index-1:] for continuation
    logits_for_continuation = logits[:, continuation_start_index - 1:-1, :]
    continuation_token_ids = input_ids[:, continuation_start_index:]
    
    print(f"Logits shape: {logits_for_continuation.shape}")
    print(f"Continuation token IDs shape: {continuation_token_ids.shape}")
    
    if continuation_token_ids.shape[1] == 0:
        print("WARNING: No continuation tokens!")
        return 0.0
    
    # Decode tokens for verification
    continuation_tokens = tokenizer.convert_ids_to_tokens(continuation_token_ids[0])
    print(f"Continuation tokens: {continuation_tokens}")
    
    log_probs = F.log_softmax(logits_for_continuation, dim=2)
    true_token_log_probs = torch.gather(
        log_probs, 2, continuation_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    print(f"Log probs per token: {true_token_log_probs[0].tolist()}")
    
    average_llh = true_token_log_probs.sum() / continuation_token_ids.shape[1]
    
    print(f"Average log-likelihood: {average_llh.item():.4f}")
    
    return average_llh.item()


def test_with_example():
    """Test with a simple example from the dataset."""
    print("=" * 60)
    print("Testing Log-Likelihood Calculation")
    print("=" * 60)
    
    # Use a small model for testing
    model_name = "gpt2"
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_name, use_fast_tokenizer=True)
        
        # Test case from dataset: stripping_VPE, item 1
        input_1 = "Alex passed Bo, but not Charlie."
        input_2 = "Alex passed Bo, but Charlie didn't."
        continuation_1 = "Charlie didn't pass Bo."
        
        print("\n" + "=" * 60)
        print("Test 1: input_2 + continuation_1")
        print("=" * 60)
        llh_1 = calculate_llh(model, tokenizer, input_2, continuation_1)
        
        print("\n" + "=" * 60)
        print("Test 2: input_1 + continuation_1")
        print("=" * 60)
        llh_2 = calculate_llh(model, tokenizer, input_1, continuation_1)
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"LLH(input_2 + continuation_1): {llh_1:.4f}")
        print(f"LLH(input_1 + continuation_1): {llh_2:.4f}")
        print(f"Log odds (left - right): {llh_1 - llh_2:.4f}")
        print(f"Test passed (left > right): {llh_1 > llh_2}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_example()


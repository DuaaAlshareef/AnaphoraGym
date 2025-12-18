"""
Targeted Assessment Engine for AnaphoraGym

This script runs behavioral assessments on language models by calculating
log-likelihoods for competing linguistic interpretations.
"""
import pandas as pd
import torch
import torch.nn.functional as F
import re
import argparse
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_dataset, load_model_and_tokenizer, get_results_dir


def calculate_llh(model, tokenizer, input_text, continuation_text):
    """
    Calculate the average log-likelihood of a continuation given an input.
    
    This function computes P(continuation | input) by:
    1. Concatenating input and continuation
    2. Tokenizing both separately to find where continuation starts
    3. Computing log-probabilities for each continuation token
    4. Averaging across all continuation tokens
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text: The input text (context)
        continuation_text: The continuation text to evaluate
    
    Returns:
        float: Average log-likelihood (log P(continuation | input))
    """
    # Validate inputs
    if not isinstance(input_text, str) or not isinstance(continuation_text, str):
        return float('nan')
    
    if not input_text.strip() or not continuation_text.strip():
        return float('nan')
    
    device = model.device
    
    # Concatenate input and continuation (no space added - tokenizer handles this)
    full_text = input_text + continuation_text
    
    # Tokenize full text and input separately
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    input_only_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_only_ids.dim() == 1:
        input_only_ids = input_only_ids.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    
    # Find where continuation starts
    continuation_start_index = input_only_ids.shape[1]
    
    # Check if continuation is empty or fully covered by input
    if continuation_start_index >= input_ids.shape[1]:
        return 0.0
    
    # Extract logits for continuation tokens
    # For causal LMs: logits[i] predicts token[i+1]
    # So logits[continuation_start_index-1] predicts token[continuation_start_index]
    # We need logits from (continuation_start_index-1) to (end-1) to predict
    # tokens from continuation_start_index to end
    logits_for_continuation = logits[:, continuation_start_index - 1:-1, :]
    continuation_token_ids = input_ids[:, continuation_start_index:]
    
    # Validate we have continuation tokens
    if continuation_token_ids.shape[1] == 0 or logits_for_continuation.shape[1] == 0:
        return 0.0
    
    # Ensure shapes match (safety check)
    if logits_for_continuation.shape[1] != continuation_token_ids.shape[1]:
        # This shouldn't happen, but handle edge case
        min_len = min(logits_for_continuation.shape[1], continuation_token_ids.shape[1])
        logits_for_continuation = logits_for_continuation[:, :min_len, :]
        continuation_token_ids = continuation_token_ids[:, :min_len]
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits_for_continuation, dim=2)
    
    # Extract log-probability of the actual continuation tokens
    true_token_log_probs = torch.gather(
        log_probs, 2, continuation_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # Average log-likelihood across all continuation tokens
    average_llh = true_token_log_probs.sum() / continuation_token_ids.shape[1]
    
    return average_llh.item()


def run_assessment(model_name: str):
    """
    Run the AnaphoraGym assessment for a specific model.
    
    Args:
        model_name: Hugging Face model identifier
    """
    print(f"--- Running AnaphoraGym Assessment for: {model_name} ---")
    
    # Load dataset
    try:
        df = load_dataset()
        print(f"Successfully loaded dataset")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            model_name,
            use_fast_tokenizer=True
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return
    
    # Run experiments
    results = []
    for index, row in df.iterrows():
        print(f"=> Processing Condition: {row['condition']}, Item: {row['item']}")
        
        for i in range(1, row['n_tests'] + 1):
            test_col_name = f'test_{i}'
            test_definition = row.get(test_col_name)
            
            if pd.isna(test_definition):
                continue
            
            # Parse test definition: left_cont_idx|left_input_idx>right_cont_idx|right_input_idx
            # Format: "1|2>1|1" means:
            #   Left: continuation_1 with input_2
            #   Right: continuation_1 with input_1
            try:
                # Strip whitespace and parse
                test_def_clean = str(test_definition).strip()
                match = re.match(r'(\d+)\|(\d+)>(\d+)\|(\d+)', test_def_clean)
                if not match:
                    print(f"  [WARN] Could not parse test definition: '{test_def_clean}'")
                    continue
                left_cont_idx, left_input_idx, right_cont_idx, right_input_idx = map(
                    int, match.groups()
                )
            except (ValueError, TypeError, AttributeError) as e:
                print(f"  [WARN] Error parsing test definition '{test_definition}': {e}")
                continue
            
            # Get input and continuation texts
            left_input = row.get(f'input_{left_input_idx}')
            left_continuation = row.get(f'continuation_{left_cont_idx}')
            right_input = row.get(f'input_{right_input_idx}')
            right_continuation = row.get(f'continuation_{right_cont_idx}')
            
            # Validate all texts are strings and not NaN/empty
            texts_to_check = {
                'left_input': left_input,
                'left_continuation': left_continuation,
                'right_input': right_input,
                'right_continuation': right_continuation
            }
            
            # Check each text - skip if any are invalid
            skip_test = False
            for name, text in texts_to_check.items():
                if pd.isna(text):
                    print(f"  [WARN] Missing {name} for test {test_col_name}")
                    skip_test = True
                    break
                if not isinstance(text, str) or not text.strip():
                    print(f"  [WARN] Invalid {name} for test {test_col_name}: '{text}'")
                    skip_test = True
                    break
            
            if skip_test:
                continue
            
            # Calculate log-likelihoods
            # Left: P(continuation_left | input_left)
            # Right: P(continuation_right | input_right)
            llh_left = calculate_llh(model, tokenizer, left_input, left_continuation)
            llh_right = calculate_llh(model, tokenizer, right_input, right_continuation)
            
            # Check for NaN results
            if pd.isna(llh_left) or pd.isna(llh_right):
                print(f"  [WARN] NaN result for test {test_col_name}, skipping")
                continue
            
            # Log odds: positive means left is more likely (test passed)
            log_odds = llh_left - llh_right
            test_passed = log_odds > 0
            
            results.append({
                'model_source': model_name,
                'condition': row['condition'],
                'item': row['item'],
                'test_name': test_col_name,
                'test_definition': test_definition,
                'LLH_left': llh_left,
                'LLH_right': llh_right,
                'logOdds': log_odds,
                'test_passed': test_passed
            })
    
    # Save results
    if not results:
        print("No valid results were generated.")
        return
    
    results_df = pd.DataFrame(results)
    results_dir = get_results_dir()
    safe_model_name = model_name.replace('/', '_')
    output_filename = f"AnaphoraGym_Results_{safe_model_name}.csv"
    output_path = os.path.join(results_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults for {model_name} saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run targeted assessment for a specific language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the Hugging Face model to test."
    )
    args = parser.parse_args()
    run_assessment(model_name=args.model)


#!/usr/bin/env python3
# coding=utf-8
"""
Quick Demo of Layer-wise Anaphora Probing
Tests the probing system with a simple example.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layer_wise_probing import AnaphoraLayerProber


def demo_simple_probe():
    """
    Demonstrate layer probing with a simple anaphora example.
    """
    print("="*80)
    print("DEMO: Layer-wise Anaphora Probing")
    print("="*80)
    print()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow.")
        print("Consider using a GPU for faster processing.")
    print()
    
    # Initialize prober
    print("Loading Llama-2-7b-chat-hf...")
    prober = AnaphoraLayerProber(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device=device
    )
    print(f"✓ Model loaded with {prober.num_layers} layers\n")
    
    # Example 1: Stripping VPE (Verb Phrase Ellipsis)
    print("-"*80)
    print("Example 1: Stripping VPE")
    print("-"*80)
    
    source_text = "Alex passed Bo, but not Charlie."
    target_text = "Alex passed Bo, but not Charlie."
    correct_continuation = "Alex didn't pass Charlie."
    incorrect_continuation = "Charlie didn't pass Bo."
    
    print(f"Source: {source_text}")
    print(f"Correct resolution: {correct_continuation}")
    print(f"Incorrect resolution: {incorrect_continuation}")
    print()
    
    print("Probing all layers (this may take a few minutes)...")
    results = prober.probe_anaphora_resolution(
        source_text=source_text,
        target_text=target_text,
        correct_continuation=correct_continuation,
        incorrect_continuation=incorrect_continuation,
        anaphora_position=-1
    )
    
    print("\nResults:")
    print(results.head(10))
    print()
    
    # Find best layer
    best_layer = results.loc[results['logprob_diff'].idxmax(), 'layer']
    best_score = results['logprob_diff'].max()
    
    print(f"🎯 Best performing layer: Layer {int(best_layer)}")
    print(f"   Log-probability difference: {best_score:.4f}")
    print()
    
    # Show top 5 layers
    print("Top 5 layers:")
    top5 = results.nlargest(5, 'logprob_diff')
    for idx, row in top5.iterrows():
        print(f"  Layer {int(row['layer']):2d}: "
              f"Score = {row['logprob_diff']:6.4f}, "
              f"Accuracy = {row['accuracy']:.0%}")
    print()
    
    # Example 2: Extract layer representations
    print("-"*80)
    print("Example 2: Extracting Layer Representations")
    print("-"*80)
    
    text = "Sarah told Maria that she would come."
    print(f"Text: {text}")
    print()
    
    print("Extracting representations from all layers...")
    layer_reps = prober.extract_layer_representations(text, position=-1)
    
    print(f"✓ Extracted {len(layer_reps)} layer representations")
    print(f"  Shape of each representation: {layer_reps[0].shape}")
    print()
    
    # Show representation statistics
    print("Representation statistics by layer:")
    print(f"{'Layer':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*60)
    
    for layer in [0, 10, 20, 30, 31]:  # Sample of layers
        rep = layer_reps[layer]
        print(f"{layer:<10} {rep.mean().item():<12.4f} "
              f"{rep.std().item():<12.4f} "
              f"{rep.min().item():<12.4f} "
              f"{rep.max().item():<12.4f}")
    
    print()
    print("="*80)
    print("Demo Complete!")
    print("="*80)
    print()
    print("To run full analysis on the dataset:")
    print("  bash scripts/mechanistic_analysis/run_layer_probing.sh")
    print()
    print("Or manually:")
    print("  python scripts/mechanistic_analysis/layer_wise_probing.py \\")
    print("      --dataset dataset/AnaphoraGym.csv \\")
    print("      --max_samples 10")
    print()


if __name__ == "__main__":
    try:
        demo_simple_probe()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

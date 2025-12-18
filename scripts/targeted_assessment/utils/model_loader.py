"""
Shared model loading utilities for the AnaphoraGym project.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables (for Hugging Face token if needed)
load_dotenv()


def get_device():
    """
    Determine the best available device.
    
    Returns:
        torch.device: The device to use
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str,
    use_fast_tokenizer: bool = True,
    load_in_8bit: bool = False
):
    """
    Load a model and tokenizer from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier
        use_fast_tokenizer: Whether to use the fast tokenizer
        load_in_8bit: Whether to load large models in 8-bit mode (for CUDA)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    # Get Hugging Face token if available
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    
    device = get_device()
    
    # Determine if this is a large model that might benefit from 8-bit loading
    is_large_model = any(k in model_name.lower() for k in ["7b", "8b", "13b", "6b"])
    
    print(f"Loading model: {model_name}...")
    print(f"Using device: {device}")
    
    try:
        # Load model
        if torch.cuda.is_available() and is_large_model and load_in_8bit:
            print("Loading large model in 8-bit mode.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                token=hf_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token
            ).to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
            token=hf_token
        )
        
        # Configure tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model successfully loaded on {device}.")
        return model, tokenizer, device
        
    except Exception as e:
        raise RuntimeError(f"Could not load model '{model_name}'. Reason: {e}") from e


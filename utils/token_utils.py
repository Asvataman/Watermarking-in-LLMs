"""
Utilities for working with tokens for watermarking and detection.
"""

import numpy as np
import tiktoken
from typing import List, Dict, Set, Tuple, Optional

def get_tokenizer(encoding_name: str = "cl100k_base"):
    """
    Get a tokenizer by name.
    
    Args:
        encoding_name: Name of the encoding
        
    Returns:
        Tokenizer
    """
    return tiktoken.get_encoding(encoding_name)

def tokenize_text(text: str, encoding_name: str = "cl100k_base") -> List[int]:
    """
    Tokenize text using the specified encoding.
    
    Args:
        text: Text to tokenize
        encoding_name: Name of the encoding
        
    Returns:
        List of token IDs
    """
    encoder = get_tokenizer(encoding_name)
    return encoder.encode(text)

def detokenize(tokens: List[int], encoding_name: str = "cl100k_base") -> str:
    """
    Convert tokens back to text.
    
    Args:
        tokens: List of token IDs
        encoding_name: Name of the encoding
        
    Returns:
        Decoded text
    """
    encoder = get_tokenizer(encoding_name)
    return encoder.decode(tokens)

def create_token_windows(
    tokens: List[int], 
    window_size: int = 100, 
    stride: int = 50
) -> List[List[int]]:
    """
    Create sliding windows of tokens.
    
    Args:
        tokens: List of token IDs
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        List of token windows
    """
    # If tokens is shorter than window_size, just return the whole list
    if len(tokens) <= window_size:
        return [tokens]
    
    windows = []
    for i in range(0, len(tokens) - window_size + 1, stride):
        windows.append(tokens[i:i + window_size])
    
    # Add the final window if it doesn't align with the stride
    if (len(tokens) - window_size) % stride != 0:
        windows.append(tokens[-window_size:])
    
    return windows

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the text.
    
    Args:
        text: Text to count tokens in
        encoding_name: Name of the encoding
        
    Returns:
        Number of tokens
    """
    encoder = get_tokenizer(encoding_name)
    tokens = encoder.encode(text)
    return len(tokens)

def estimate_token_usage(messages: List[Dict[str, str]]) -> int:
    """
    Estimate the number of tokens used by a list of messages.
    
    Args:
        messages: List of messages in the format {"role": role, "content": content}
        
    Returns:
        Estimated number of tokens
    """
    encoder = get_tokenizer()
    
    # Base tokens per message
    tokens_per_message = 4
    
    # Base tokens for the entire request
    tokens_per_request = 3
    
    total_tokens = tokens_per_request
    
    for message in messages:
        # Add tokens for each message
        total_tokens += tokens_per_message
        
        # Add tokens for the content
        total_tokens += len(encoder.encode(message["content"]))
        
        # Add tokens for the role
        total_tokens += len(encoder.encode(message["role"]))
    
    return total_tokens
"""
Utilities for processing text for watermarking and detection.
"""

import re
from typing import List, Dict, Tuple, Optional

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Text to extract code blocks from
        
    Returns:
        List of tuples (language, code)
    """
    # Match code blocks with language specification
    pattern = r"```(\w*)\n([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    return matches

def remove_code_block_markers(text: str) -> str:
    """
    Remove code block markers from text but keep the code.
    
    Args:
        text: Text with code blocks
        
    Returns:
        Text with code block markers removed
    """
    # Remove the opening markers
    text = re.sub(r"```\w*\n", "", text)
    
    # Remove the closing markers
    text = re.sub(r"```", "", text)
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    
    # Remove spaces at the beginning and end of lines
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting (can be improved with NLP libraries)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    return [s.strip() for s in sentences if s.strip()]

def handle_special_formatting(text: str) -> str:
    """
    Handle special formatting like code blocks for watermark detection.
    
    Args:
        text: Text to process
        
    Returns:
        Processed text
    """
    # Handle code blocks
    text = remove_code_block_markers(text)
    
    # Handle inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    
    # Handle bold and italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    
    # Handle links
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    
    return text

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

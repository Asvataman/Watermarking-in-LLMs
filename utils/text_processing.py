import re
from typing import List, Dict, Tuple, Optional

def preprocess_text(text: str) -> str:
    """
    Preprocess text for watermarking/detection.
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text
import numpy as np
import hashlib
import tiktoken
from scipy import stats
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import io
import base64
import streamlit as st

class WatermarkDetector:
    """
    Simplified detector for the watermarking technique described in "A Watermark for Large Language Models"
    by Kirchenbauer et al.
    
    This implementation uses only the immediately preceding token to determine
    the green list for the current token, matching the basic approach in the paper.
    """
    
    def __init__(
        self,
        gamma: float = 0.5,
        seed: int = 42,
        threshold: float = 4.0,
        tokenizer_name: str = "cl100k_base"  # Default for GPT-4 models
    ):
        """
        Initialize the watermark detector.
        
        Args:
            gamma: Proportion of tokens in the "green list" (0-1)
            seed: Seed for the hash function
            threshold: Z-score threshold for detection
            tokenizer_name: Tiktoken encoding to use for tokenization
        """
        self.gamma = gamma
        self.seed = seed
        self.threshold = threshold
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab
        
    def _get_green_tokens(self, prev_token: int) -> List[int]:
        """
        Generate the "green list" of tokens based on the single previous token.
        
        Args:
            prev_token: Previous token ID
            
        Returns:
            List of token IDs in the green list
        """
        # Create a deterministic hash from the previous token
        hash_input = f"{prev_token}_{self.seed}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        
        # Use the hash to seed the random number generator
        # Ensure seed is within valid range (0 to 2^32 - 1)
        seed_value = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**32 - 1)
        rng = np.random.RandomState(seed_value)
        
        # Randomly select tokens for the green list
        num_green = int(self.gamma * self.vocab_size)
        green_tokens = rng.choice(self.vocab_size, size=num_green, replace=False)
        
        return green_tokens.tolist()
    
    def detect(self, text: str) -> Dict:
        """
        Detect whether a text has been watermarked.
        
        Args:
            text: Text to check for watermark
            
        Returns:
            Dictionary with detection results
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        # Require a minimum number of tokens for reliable detection
        min_tokens = 25
        if len(tokens) < min_tokens:
            return {
                "is_watermarked": False,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_tokens": [],
                "green_indices": [],
                "total_tokens": len(tokens),
                "green_token_count": 0,
                "expected_green_count": len(tokens) * self.gamma,
                "visualization": None
            }
        
        # Track which tokens are in the green list
        green_indices = []
        green_tokens = []
        
        # For debugging
        print("Starting detection with simplified approach")
        print(f"Text length in tokens: {len(tokens)}")
        
        # Check each token, starting from the second token
        # (since we need the previous token to generate the green list)
        for i in range(1, len(tokens)):
            prev_token = tokens[i-1]  # Only use the immediately preceding token
            current_token = tokens[i]
            
            # Get green list for this position
            green_list = self._get_green_tokens(prev_token)
            
            # Check if current token is in the green list
            if current_token in green_list:
                green_indices.append(i)
                green_tokens.append(current_token)
                
            # Debug for first few tokens
            if i < 5:
                print(f"Token {i}: prev={prev_token}, current={current_token}, is_green={current_token in green_list}")
        
        # Calculate statistics
        total_tokens = len(tokens) - 1  # Exclude the first token
        green_token_count = len(green_indices)
        expected_green_count = total_tokens * self.gamma
        std_dev = np.sqrt(total_tokens * self.gamma * (1 - self.gamma))
        
        # Calculate z-score
        if std_dev > 0:
            z_score = (green_token_count - expected_green_count) / std_dev
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 1.0
            
        print(f"Detection results: green={green_token_count}, expected={expected_green_count}, z-score={z_score}")
        
        # Generate visualization
        visualization = self._generate_visualization(tokens, green_indices, z_score, p_value)
        
        # Return detection results
        return {
            "is_watermarked": z_score > self.threshold,
            "z_score": z_score,
            "p_value": p_value,
            "green_tokens": green_tokens,
            "green_indices": green_indices,
            "total_tokens": total_tokens,
            "green_token_count": green_token_count,
            "expected_green_count": expected_green_count,
            "visualization": visualization
        }
    
    def _generate_visualization(self, tokens, green_indices, z_score, p_value):
        """Generate a visualization of the watermark detection."""
        plt.figure(figsize=(10, 6))
        
        # Create bar chart of expected vs actual green tokens
        total_tokens = len(tokens) - 1
        green_token_count = len(green_indices)
        expected_green_count = total_tokens * self.gamma
        
        plt.bar(['Expected Green Tokens', 'Actual Green Tokens'], 
                [expected_green_count, green_token_count],
                color=['lightgray', 'green'])
        
        plt.axhline(y=expected_green_count + self.threshold * np.sqrt(total_tokens * self.gamma * (1 - self.gamma)), 
                   color='red', linestyle='--', label=f'Detection Threshold (z={self.threshold})')
        
        plt.title(f'Watermark Detection Results\nz-score: {z_score:.2f}, p-value: {p_value:.8f}')
        plt.ylabel('Token Count')
        plt.legend()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def get_highlighted_text(self, text: str, detection_result: Dict) -> str:
        """
        Get HTML with green tokens highlighted for visualization.
        
        Args:
            text: Original text
            detection_result: Detection result from detect()
            
        Returns:
            HTML string with green tokens highlighted
        """
        tokens = self.tokenizer.encode(text)
        green_indices = detection_result["green_indices"]
        
        # Convert tokens back to text with highlighting
        html_parts = []
        
        for i in range(len(tokens)):
            token_text = self.tokenizer.decode([tokens[i]])
            
            if i in green_indices:
                html_parts.append(f'<span style="background-color: #AAFFAA;">{token_text}</span>')
            else:
                html_parts.append(token_text)
        
        return "".join(html_parts)
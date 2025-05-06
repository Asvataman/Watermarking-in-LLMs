import numpy as np
import hashlib
import tiktoken
import os
from typing import List, Dict, Tuple, Optional

class Watermarker:
    """
    Implementation of the watermarking technique described in "A Watermark for Large Language Models"
    by Kirchenbauer et al.
    
    This class adds a "soft watermark" to LLM outputs by biasing the logits of a random subset of tokens
    during generation.
    """
    
    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        context_width: int = 1,
        seed: int = 42,
        tokenizer_name: str = "cl100k_base"  # Default for GPT-4 models
    ):
        """
        Initialize the watermarker.
        
        Args:
            gamma: Proportion of tokens to include in the "green list" (0-1)
            delta: Logit bias to add to green list tokens
            context_width: Number of previous tokens to use for generating the hash
            seed: Seed for the hash function
            tokenizer_name: Tiktoken encoding to use for tokenization
        """
        self.gamma = gamma
        self.delta = delta
        self.context_width = context_width
        self.seed = seed
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab
        
    def _get_green_tokens(self, prev_tokens: List[int]) -> List[int]:
        """
        Generate the "green list" of tokens based on previous tokens.
        
        Args:
            prev_tokens: List of previous token IDs
            
        Returns:
            List of token IDs in the green list
        """
        # Only use the most recent context_width tokens
        context = prev_tokens[-self.context_width:]
        
        # Create a deterministic hash from the previous tokens
        context_str = ",".join([str(t) for t in context])
        hash_input = f"{context_str}_{self.seed}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        
        # Use the hash to seed the random number generator
        # Ensure seed is within valid range (0 to 2^32 - 1)
        seed_value = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**32 - 1)
        rng = np.random.RandomState(seed_value)
        
        # Randomly select tokens for the green list
        num_green = int(self.gamma * self.vocab_size)
        green_tokens = rng.choice(self.vocab_size, size=num_green, replace=False)
        
        return green_tokens.tolist()
    
    def prepare_logit_biases(self, prev_tokens: List[int]) -> Dict[int, float]:
        """
        Prepare the logit biases for the next token generation.
        
        Args:
            prev_tokens: List of previous token IDs
            
        Returns:
            Dictionary mapping token IDs to their logit biases
        """
        if not prev_tokens:
            return {}
            
        green_tokens = self._get_green_tokens(prev_tokens)
        
        # Azure OpenAI API limits logit_bias to 300 entries (stricter than regular OpenAI API)
        max_logit_bias_entries = 300
        
        # If green list is larger than API limit, randomly sample tokens
        if len(green_tokens) > max_logit_bias_entries:
            # Generate a seed that's within the valid range (0 to 2^32 - 1)
            seed_value = (self.seed + hash(str(prev_tokens[-1]))) % (2**32 - 1)
            rng = np.random.RandomState(int(seed_value))
            green_tokens = rng.choice(green_tokens, size=max_logit_bias_entries, replace=False).tolist()
        
        # Create a dictionary of logit biases for the green tokens
        logit_biases = {token_id: self.delta for token_id in green_tokens}
        
        print(f"Applied watermark with {len(logit_biases)} biased tokens (max allowed: 300)")
        return logit_biases
    
    def generate_with_watermark(self, client, messages, max_tokens=500, **kwargs):
        """
        Generate text with the Azure OpenAI API while applying the watermark.
        
        Args:
            client: Azure OpenAI API client
            messages: List of message dictionaries to send to the API
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Generated response with watermark
        """
        # Initialize with the system message if present
        prev_tokens = []
        if messages and messages[0]['role'] == 'system':
            system_tokens = self.tokenizer.encode(messages[0]['content'])
            prev_tokens.extend(system_tokens)
        
        # Add all user/assistant messages
        for message in messages:
            if message['role'] != 'system':  # Skip system message (already processed)
                msg_tokens = self.tokenizer.encode(message['content'])
                prev_tokens.extend(msg_tokens)
        
        # Prepare logit biases
        logit_biases = self.prepare_logit_biases(prev_tokens)
        
        try:
            # Generate with watermark
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                logit_bias=logit_biases,
                **kwargs
            )
            print("Successfully generated watermarked response")
            return response
        except Exception as e:
            print(f"Error in watermarked generation: {e}")
            print(f"Falling back to generation without watermark")
            
            # Fall back to generation without watermark
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
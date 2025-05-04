import numpy as np
import hashlib
import tiktoken
from typing import List, Dict, Tuple, Set, Optional
from config.setting import DEFAULT_GAMMA, DEFAULT_DELTA, DEFAULT_SEED, VOCABULARY_SIZE

class Watermarker:
    """
    Implements the watermarking technique described in "A Watermark for Large Language Models"
    by Kirchenbauer et al. (2023).
    
    The watermarking algorithm works by:
    1. Using a secret key to divide the vocabulary into "green" and "red" tokens
    2. Biasing the model's output distribution to slightly prefer "green" tokens
    3. Statistically analyzing token distributions to detect the watermark
    """
    
    def __init__(
        self, 
        vocabulary_size: int = VOCABULARY_SIZE,
        gamma: float = DEFAULT_GAMMA,
        delta: float = DEFAULT_DELTA,
        seed: Optional[int] = DEFAULT_SEED,
        encoding_name: str = "cl100k_base"  # GPT-4 tokenizer
    ):
        """
        Initialize the watermarker.
        
        Args:
            vocabulary_size: Size of the model's vocabulary
            gamma: Fraction of tokens to mark as "green" (watermark tokens)
            delta: Logit bias to apply to green tokens
            seed: Random seed for watermark key
            encoding_name: Name of the tokenizer encoding
        """
        self.vocabulary_size = vocabulary_size
        self.gamma = gamma
        self.delta = delta
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Secret key for the watermark
        self.watermark_key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generate a secret key for the watermark."""
        return hashlib.sha256(str(self.seed).encode()).digest()
    
    def _get_green_tokens(self, prev_tokens: List[int]) -> Set[int]:
        """
        Determine which tokens are "green" (watermark tokens) based on previous tokens.
        
        Args:
            prev_tokens: List of token IDs of previously generated tokens
        
        Returns:
            Set of token IDs marked as "green"
        """
        if not prev_tokens:
            # Use default seed if no previous context
            hash_input = self.watermark_key
        else:
            # Use the last token (or a window of tokens) to seed the hash
            context = str(prev_tokens[-1]).encode()
            hash_input = hashlib.sha256(self.watermark_key + context).digest()
        
        # Use the hash to seed a random number generator
        rng = np.random.RandomState(int.from_bytes(hash_input[:4], byteorder='big'))
        
        # Randomly select gamma fraction of tokens as "green"
        green_size = int(self.gamma * self.vocabulary_size)
        green_tokens = set(rng.choice(self.vocabulary_size, size=green_size, replace=False))
        
        return green_tokens
    
    def apply_watermark(self, logits: np.ndarray, prev_tokens: List[int]) -> np.ndarray:
        """
        Apply the watermark by modifying the logits for green tokens.
        
        Args:
            logits: Original model logits (vocabulary size)
            prev_tokens: List of token IDs of previously generated tokens
            
        Returns:
            Modified logits with watermark applied
        """
        green_tokens = self._get_green_tokens(prev_tokens)
        modified_logits = logits.copy()
        
        # Apply the bias to green tokens
        for token_id in green_tokens:
            if token_id < len(modified_logits):
                modified_logits[token_id] += self.delta
                
        return modified_logits
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        return self.encoding.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        return self.encoding.decode(tokens)

    def get_watermark_status(self) -> Dict:
        """Return the current watermark parameters as a dictionary."""
        return {
            "enabled": True,
            "gamma": self.gamma,
            "delta": self.delta,
            "seed": self.seed
        }


class AzureWatermarkedClient:
    """
    Wrapper around Azure OpenAI client that applies watermarking to the generated text.
    """
    
    def __init__(self, azure_client, watermarker: Watermarker):
        """
        Initialize the watermarked client.
        
        Args:
            azure_client: Azure OpenAI client
            watermarker: Watermarker instance
        """
        self.azure_client = azure_client
        self.watermarker = watermarker
        self.watermarking_enabled = True
        
    def toggle_watermarking(self, enabled: bool = True):
        """Toggle watermarking on or off."""
        self.watermarking_enabled = enabled
        
    async def generate_completion(self, messages, **kwargs):
        """
        Generate a completion with watermarking applied.
        
        This is a simplified implementation. In practice, you would need to
        integrate with the Azure OpenAI client's tokenizer and modify the
        sampling distribution directly, which may require custom API access.
        
        For this POC, we apply a logit bias to implement the watermark, which is
        a simplified version of the approach described in the paper.
        """
        if not self.watermarking_enabled:
            # If watermarking is disabled, just pass through to the Azure client
            return await self.azure_client.get_chat_response(
                messages=messages,
                **kwargs
            )
        
        # For watermarking, we need to identify green tokens and apply logit biases
        # Extract the last message to determine context
        last_message_content = messages[-1]["content"] if messages else ""
        prev_tokens = self.watermarker.tokenize(last_message_content)
        
        # Get green tokens for this context
        green_tokens = self.watermarker._get_green_tokens(prev_tokens)
        
        # Convert to logit_bias format for OpenAI API
        # This is a simplified approach; the paper's method would require more direct access
        logit_bias = {}
        for token in green_tokens:
            if token < self.watermarker.vocabulary_size:
                logit_bias[str(token)] = self.watermarker.delta
        
        # Add logit bias to kwargs
        if "logit_bias" in kwargs:
            kwargs["logit_bias"].update(logit_bias)
        else:
            kwargs["logit_bias"] = logit_bias
        
        # Generate completion with the biased distribution
        return await self.azure_client.get_chat_response(
            messages=messages,
            **kwargs
        )
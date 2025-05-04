import numpy as np
import hashlib
import re
from scipy import stats
from typing import List, Dict, Tuple, Set, Optional, Union
import tiktoken

class WatermarkDetector:
    """
    Implements the watermark detection algorithm described in "A Watermark for Large Language Models"
    by Kirchenbauer et al. (2023).
    
    The detection works by:
    1. Tokenizing the input text
    2. Computing the same green/red token partition using the secret key
    3. Performing statistical analysis to determine if green tokens are overrepresented
    """
    
    def __init__(
        self, 
        vocabulary_size: int = 100000,
        gamma: float = 0.5,
        z_threshold: float = 4.0,  # z-score threshold for detection
        min_tokens: int = 30,  # Minimum number of tokens required for reliable detection
        seed: Optional[int] = None,
        encoding_name: str = "cl100k_base"  # GPT-4 tokenizer
    ):
        """
        Initialize the watermark detector.
        
        Args:
            vocabulary_size: Size of the model's vocabulary
            gamma: Fraction of tokens marked as "green" (must match watermarker)
            z_threshold: Z-score threshold for detection (higher = fewer false positives)
            min_tokens: Minimum number of tokens required for reliable detection
            seed: Random seed for watermark key (must match watermarker)
            encoding_name: Name of the tokenizer encoding
        """
        self.vocabulary_size = vocabulary_size
        self.gamma = gamma
        self.z_threshold = z_threshold
        self.min_tokens = min_tokens
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Secret key for the watermark
        self.watermark_key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generate a secret key for the watermark."""
        return hashlib.sha256(str(self.seed).encode()).digest()
    
    def _get_green_tokens(self, prev_token: int) -> Set[int]:
        """
        Determine which tokens are "green" (watermark tokens) based on the previous token.
        
        Args:
            prev_token: Token ID of the previous token
        
        Returns:
            Set of token IDs marked as "green"
        """
        # Use the previous token to seed the hash
        context = str(prev_token).encode()
        hash_input = hashlib.sha256(self.watermark_key + context).digest()
        
        # Use the hash to seed a random number generator
        rng = np.random.RandomState(int.from_bytes(hash_input[:4], byteorder='big'))
        
        # Randomly select gamma fraction of tokens as "green"
        green_size = int(self.gamma * self.vocabulary_size)
        green_tokens = set(rng.choice(self.vocabulary_size, size=green_size, replace=False))
        
        return green_tokens
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to handle special cases like code formatting.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Handle code blocks with formatting
        # Remove markdown code block syntax but keep the code
        text = re.sub(r'```[a-zA-Z0-9]*\n', '', text)
        text = re.sub(r'```', '', text)
        
        # Handle inline code formatting
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        return text
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        preprocessed_text = self._preprocess_text(text)
        return self.encoding.encode(preprocessed_text)
    
    def detect_watermark(self, text: str) -> Dict:
        """
        Detect if the text contains a watermark.
        
        Args:
            text: Text to analyze for watermark
            
        Returns:
            Dictionary with detection results including:
            - is_watermarked: Boolean indicating if watermark is detected
            - z_score: Z-score of the statistical test
            - p_value: P-value of the statistical test
            - green_count: Number of green tokens detected
            - red_count: Number of red tokens detected
            - total_tokens: Total number of tokens analyzed
            - confidence: Detection confidence level
        """
        tokens = self.tokenize(text)
        
        if len(tokens) < self.min_tokens:
            return {
                "is_watermarked": False,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_count": 0,
                "red_count": 0,
                "total_tokens": len(tokens),
                "confidence": 0.0,
                "error": f"Text too short for reliable detection ({len(tokens)} tokens, minimum {self.min_tokens})"
            }
        
        green_count = 0
        
        # For each token, determine if the next token is in the green list
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            
            green_tokens = self._get_green_tokens(current_token)
            if next_token in green_tokens:
                green_count += 1
        
        # Calculate expected green count under null hypothesis (no watermark)
        total_analyzed = len(tokens) - 1  # Number of token pairs analyzed
        expected_green = total_analyzed * self.gamma
        expected_std = np.sqrt(total_analyzed * self.gamma * (1 - self.gamma))
        
        # Calculate z-score
        z_score = (green_count - expected_green) / expected_std if expected_std > 0 else 0
        
        # Calculate p-value (one-tailed test)
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Determine if the text is watermarked
        is_watermarked = z_score > self.z_threshold
        
        # Calculate confidence level (0-100%)
        confidence = min(100, max(0, (z_score / self.z_threshold) * 100)) if z_score > 0 else 0
        
        return {
            "is_watermarked": is_watermarked,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "green_count": green_count,
            "red_count": total_analyzed - green_count,
            "total_tokens": len(tokens),
            "analyzed_pairs": total_analyzed,
            "confidence": float(confidence)
        }
    
    def analyze_text_segments(self, text: str, window_size: int = 100, stride: int = 50) -> Dict:
        """
        Analyze text using a sliding window approach to locate watermarked segments.
        
        Args:
            text: Text to analyze
            window_size: Number of tokens in each analysis window
            stride: Number of tokens to move the window by
            
        Returns:
            Dictionary with overall and per-segment analysis
        """
        tokens = self.tokenize(text)
        segments = []
        
        # If text is shorter than window_size, just analyze the whole text
        if len(tokens) <= window_size:
            result = self.detect_watermark(text)
            segments.append({
                "start_token": 0,
                "end_token": len(tokens),
                "text": text,
                **result
            })
            
            return {
                "overall": result,
                "segments": segments
            }
        
        # Analyze text using sliding window
        for start in range(0, len(tokens) - window_size, stride):
            end = start + window_size
            segment_tokens = tokens[start:end]
            segment_text = self.encoding.decode(segment_tokens)
            
            result = self.detect_watermark(segment_text)
            segments.append({
                "start_token": start,
                "end_token": end,
                "text": segment_text,
                **result
            })
        
        # Also analyze the final segment if not covered
        if segments[-1]["end_token"] < len(tokens):
            start = len(tokens) - window_size
            segment_tokens = tokens[start:]
            segment_text = self.encoding.decode(segment_tokens)
            
            result = self.detect_watermark(segment_text)
            segments.append({
                "start_token": start,
                "end_token": len(tokens),
                "text": segment_text,
                **result
            })
        
        # Calculate overall result
        overall_green = sum(seg["green_count"] for seg in segments)
        overall_analyzed = sum(seg["analyzed_pairs"] for seg in segments)
        
        if overall_analyzed == 0:
            overall_result = {
                "is_watermarked": False,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_count": 0,
                "red_count": 0,
                "total_tokens": len(tokens),
                "confidence": 0.0,
                "error": "No tokens could be analyzed"
            }
        else:
            expected_green = overall_analyzed * self.gamma
            expected_std = np.sqrt(overall_analyzed * self.gamma * (1 - self.gamma))
            
            z_score = (overall_green - expected_green) / expected_std if expected_std > 0 else 0
            p_value = 1 - stats.norm.cdf(z_score)
            
            is_watermarked = z_score > self.z_threshold
            confidence = min(100, max(0, (z_score / self.z_threshold) * 100)) if z_score > 0 else 0
            
            overall_result = {
                "is_watermarked": is_watermarked,
                "z_score": float(z_score),
                "p_value": float(p_value),
                "green_count": overall_green,
                "red_count": overall_analyzed - overall_green,
                "total_tokens": len(tokens),
                "analyzed_pairs": overall_analyzed,
                "confidence": float(confidence)
            }
        
        return {
            "overall": overall_result,
            "segments": segments
        }

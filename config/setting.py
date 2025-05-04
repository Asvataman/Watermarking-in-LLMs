"""
Global settings for the LLM watermarking application.
"""

# Azure OpenAI settings
AZURE_API_VERSION = "2023-12-01-preview"

# Watermarking settings
DEFAULT_GAMMA = 0.5  # Fraction of tokens to mark as "green"
DEFAULT_DELTA = 2.0  # Logit bias for green tokens
DEFAULT_SEED = 42  # Random seed for watermarking
DEFAULT_Z_THRESHOLD = 4.0  # Z-score threshold for detection

# Tokenizer settings
TOKENIZER_NAME = "cl100k_base"  # GPT-4 tokenizer
VOCABULARY_SIZE = 500  # Approximate vocabulary size

# Detection settings
MIN_TOKENS_FOR_DETECTION = 30  # Minimum number of tokens needed for reliable detection
DETECTION_WINDOW_SIZE = 100  # Number of tokens in sliding window for analysis
DETECTION_WINDOW_STRIDE = 50  # Step size for sliding window
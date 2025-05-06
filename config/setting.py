import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI API settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

# Watermarking parameters
# gamma: proportion of vocabulary in green list (typically 0.5 or 0.25)
# delta: logit bias for green list tokens (typically 2.0 for good detection)
WATERMARK_GAMMA = float(os.getenv("WATERMARK_GAMMA", "0.5"))  # Proportion of vocabulary in green list
WATERMARK_DELTA = float(os.getenv("WATERMARK_DELTA", "2"))  # Increased default value for stronger watermarking

# Detection threshold for watermark (z-score)
# Paper suggests 4.0 for good balance between detection and false positives
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))

# Seed for watermarking hash function
# In production, this should be kept private if using private watermarking
WATERMARK_SEED = int(os.getenv("WATERMARK_SEED", "42"))
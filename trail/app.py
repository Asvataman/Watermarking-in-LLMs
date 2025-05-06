"""
LLM Watermarking Streamlit Application

This application provides a user interface for:
1. Generating text with or without watermarks
2. Detecting watermarks in text with visual highlighting
"""

import streamlit as st
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import hashlib
from typing import List, Tuple, Dict, Set, Optional
import torch.nn.functional as F
from tqdm import tqdm
import re
import time

# Import the LLMWatermarker class (assumed to be in a file called watermarker.py)
# For this demo, we're including the class directly

class LLMWatermarker:
    """
    A class that implements watermarking for Large Language Models.
    
    The watermarking scheme works by:
    1. Using a keyed hash function to deterministically partition vocabulary tokens
       into "green" (preferred) and "red" (avoided) lists for each generation step
    2. Biasing the model's sampling during generation to prefer green-list tokens
    3. Detecting watermarks by measuring the statistical bias toward green tokens
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        gamma: float = 0.5,  # Watermark strength parameter
        delta: float = 2.0,  # Detection threshold parameter
        seeding_scheme: str = "hash_prev_tokens",  # How to generate the random seed
        hash_key: str = "my_secret_key",  # Secret key for hash function
        green_list_fraction: float = 0.5,  # Fraction of tokens in the green list
        n_prev_tokens: int = 4,  # Number of previous tokens to use for seeding
    ):
        """
        Initialize the watermarker with the specified parameters.
        
        Args:
            model_name: HuggingFace model identifier
            gamma: Strength of the watermark bias (higher = stronger bias)
            delta: Z-score threshold for detection
            seeding_scheme: Method to generate seeds for token partitioning
            hash_key: Secret key for the hash function
            green_list_fraction: Fraction of tokens to include in the green list
            n_prev_tokens: Number of previous tokens to use for context hash
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.vocab_size = len(self.tokenizer)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key
        self.green_list_fraction = green_list_fraction
        self.n_prev_tokens = n_prev_tokens
        
        st.sidebar.write(f"Model: {model_name}")
        st.sidebar.write(f"Vocabulary size: {self.vocab_size}")
        
    def _get_seed(self, prev_tokens: List[int]) -> int:
        """
        Generate a seed for the token partitioning based on previous tokens.
        
        Args:
            prev_tokens: List of previous token IDs
            
        Returns:
            A deterministic seed value
        """
        if self.seeding_scheme == "hash_prev_tokens":
            # Use the last n_prev_tokens for context
            context = prev_tokens[-self.n_prev_tokens:] if len(prev_tokens) >= self.n_prev_tokens else prev_tokens
            
            # Create a deterministic seed from the context and secret key
            context_str = ",".join([str(t) for t in context])
            hash_input = f"{self.hash_key}:{context_str}"
            hash_output = hashlib.sha256(hash_input.encode()).hexdigest()
            
            # Convert hash to integer seed
            seed = int(hash_output, 16) % (2**32)
            return seed
        
        elif self.seeding_scheme == "fixed":
            # Just use a fixed seed (not recommended for production)
            return 42
        
        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")
    
    def _partition_vocabulary(self, seed: int) -> Tuple[Set[int], Set[int]]:
        """
        Partition the vocabulary into green list and red list based on the seed.
        
        Args:
            seed: Random seed for the partitioning
            
        Returns:
            A tuple of (green_list, red_list) as sets of token IDs
        """
        # Set the random seed for reproducibility
        np.random.seed(seed)
        
        # Create a random permutation of the vocabulary
        permutation = np.random.permutation(self.vocab_size)
        
        # Determine the size of the green list
        green_size = int(self.vocab_size * self.green_list_fraction)
        
        # Partition the vocabulary
        green_list = set(permutation[:green_size])
        red_list = set(permutation[green_size:])
        
        return green_list, red_list
    
    def _apply_watermark_bias(
        self, 
        logits: torch.Tensor, 
        green_list: Set[int]
    ) -> torch.Tensor:
        """
        Apply the watermark bias to the model's logits.
        
        Args:
            logits: Model output logits
            green_list: Set of token IDs in the green list
            
        Returns:
            Watermarked logits
        """
        # Convert green_list to a tensor mask
        green_mask = torch.zeros_like(logits, dtype=torch.bool)
        green_indices = list(green_list)
        green_mask[0, green_indices] = True
        
        # Apply the bias: add gamma to green-list tokens
        biased_logits = logits.clone()
        biased_logits[green_mask] += self.gamma
        
        return biased_logits
    
    def generate_watermarked_text(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 1.0,
        use_watermark: bool = True,
        progress_callback=None
    ) -> Tuple[str, List[int], List[Set[int]]]:
        """
        Generate text with a watermark.
        
        Args:
            prompt: The starting text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            use_watermark: Whether to apply the watermark
            progress_callback: Optional callback for progress updates
            
        Returns:
            A tuple of (generated_text, token_ids, green_lists)
        """
        self.model.eval()
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        cumulative_tokens = input_ids[0].tolist()
        
        # Store green lists for verification later
        all_green_lists = []
        
        # Generation loop
        for i in range(max_length):
            if progress_callback:
                progress_callback(i, max_length)
                
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            if use_watermark:
                # Get the seed from previous tokens
                seed = self._get_seed(cumulative_tokens)
                
                # Partition vocabulary
                green_list, _ = self._partition_vocabulary(seed)
                
                # Save green list for verification
                all_green_lists.append(green_list)
                
                # Apply watermark bias
                logits = self._apply_watermark_bias(logits, green_list)
            else:
                # Still track what would have been the green lists
                seed = self._get_seed(cumulative_tokens)
                green_list, _ = self._partition_vocabulary(seed)
                all_green_lists.append(green_list)
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            cumulative_tokens.append(next_token.item())
            
            # Stop if we generate an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            # Generate intermediate result for display
            if progress_callback and i % 5 == 0:
                intermediate_text = self.tokenizer.decode(cumulative_tokens, skip_special_tokens=True)
                progress_callback(i, max_length, intermediate_text)
        
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(cumulative_tokens, skip_special_tokens=True)
        
        return generated_text, cumulative_tokens, all_green_lists
    
    def detect_watermark(
        self, 
        text: str,
        z_score_threshold: Optional[float] = None,
        return_token_colors: bool = False
    ) -> Dict:
        """
        Detect if text contains a watermark.
        
        Args:
            text: The text to analyze
            z_score_threshold: Custom threshold (uses self.delta if None)
            return_token_colors: Whether to return token-level color information
            
        Returns:
            Dictionary with detection results and optional token colors
        """
        if z_score_threshold is None:
            z_score_threshold = self.delta
            
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        token_text_mapping = []
        
        # Get the text corresponding to each token for visualization
        decoded_tokens = []
        for i, token_id in enumerate(tokens):
            if i == 0:
                token_text = self.tokenizer.decode([token_id])
            else:
                # Get text for this token
                token_text = self.tokenizer.decode([token_id])
                # In some cases, we need to handle whitespace differently
                if not token_text.startswith(' ') and i > 0:
                    prev_text = self.tokenizer.decode(tokens[:i])
                    current_text = self.tokenizer.decode(tokens[:i+1])
                    token_text = current_text[len(prev_text):]
            
            decoded_tokens.append(token_text)
            
        if len(tokens) <= self.n_prev_tokens:
            return {
                "has_watermark": False,
                "z_score": 0.0,
                "p_value": 0.5,
                "green_fraction": 0.0,
                "message": "Text too short for reliable detection",
                "token_colors": []
            }
        
        # Count green tokens
        green_count = 0
        token_colors = []  # Will contain 'green', 'red', or 'neutral'
        
        # Skip the first few tokens as they won't have enough context
        for i in range(len(tokens)):
            if i < self.n_prev_tokens:
                token_colors.append("neutral")
                continue
                
            prev_tokens = tokens[:i]
            current_token = tokens[i]
            
            # Compute the seed and green list
            seed = self._get_seed(prev_tokens)
            green_list, red_list = self._partition_vocabulary(seed)
            
            # Check if the current token is in the green list
            if current_token in green_list:
                green_count += 1
                token_colors.append("green")
            else:
                token_colors.append("red")
        
        # Statistical analysis
        n = len(tokens) - self.n_prev_tokens  # Number of analyzed tokens
        observed_green_fraction = green_count / n if n > 0 else 0
        expected_green_fraction = self.green_list_fraction
        
        # Standard deviation under null hypothesis (binomial distribution)
        std_dev = np.sqrt(expected_green_fraction * (1 - expected_green_fraction) / n) if n > 0 else 1
        
        # Calculate Z-score
        z_score = (observed_green_fraction - expected_green_fraction) / std_dev if std_dev > 0 else 0
        
        # Convert to p-value (one-sided test)
        from scipy import stats
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Determine if watermark is present
        has_watermark = z_score > z_score_threshold
        
        result = {
            "has_watermark": has_watermark,
            "z_score": z_score,
            "p_value": p_value,
            "green_fraction": observed_green_fraction,
            "expected_green_fraction": expected_green_fraction,
            "num_tokens_analyzed": n,
            "green_count": green_count,
            "red_count": n - green_count,
        }
        
        if return_token_colors:
            result["token_colors"] = list(zip(decoded_tokens, token_colors))
            
        return result
    
    def visualize_detection(self, detection_result):
        """
        Create matplotlib visualization for watermark detection results.
        
        Args:
            detection_result: The result from detect_watermark
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart of green token fraction
        labels = ["Detected", "Expected", "Threshold"]
        green_fractions = [
            detection_result["green_fraction"],
            detection_result["expected_green_fraction"],
            detection_result["expected_green_fraction"] + self.delta * np.sqrt(
                self.green_list_fraction * (1 - self.green_list_fraction) / detection_result["num_tokens_analyzed"]
            )
        ]
        
        colors = ['green' if detection_result["has_watermark"] else 'red', 'blue', 'orange']
        
        x = np.arange(len(labels))
        bars = ax.bar(x, green_fractions, width=0.6, color=colors)
        
        # Customize the plot
        ax.set_ylabel('Green Token Fraction')
        ax.set_title('Watermark Detection Results')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Add z-score as text
        ax.text(0, green_fractions[0] + 0.01, 
                f'z={detection_result["z_score"]:.2f}', ha='center')
        
        # Set y-axis to start from 0 and have some headroom
        ax.set_ylim(0, max(green_fractions) * 1.2)
        
        plt.tight_layout()
        return fig


# Load watermarker model (do this once)
@st.cache_resource
def load_watermarker():
    watermarker = LLMWatermarker(
        model_name="gpt2",  # Using a small model for demo purposes
        gamma=2.0,          # Watermark strength
        delta=2.0,          # Detection threshold
        green_list_fraction=0.5,
        n_prev_tokens=4
    )
    return watermarker

def highlight_text(token_colors):
    """
    Create HTML with colored highlighting based on token colors.
    
    Args:
        token_colors: List of (token_text, color) tuples
    
    Returns:
        HTML string with highlighted text
    """
    html = ""
    for token_text, color in token_colors:
        if color == "green":
            html += f'<span style="background-color: rgba(0, 255, 0, 0.3);">{token_text}</span>'
        elif color == "red":
            html += f'<span style="background-color: rgba(255, 0, 0, 0.2);">{token_text}</span>'
        else:  # neutral
            html += f'<span style="color: gray;">{token_text}</span>'
    
    return html

def main():
    st.set_page_config(
        page_title="LLM Watermarking Tool",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Load model
    with st.spinner("Loading model... (this may take a moment)"):
        watermarker = load_watermarker()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    gamma = st.sidebar.slider("Watermark Strength (γ)", 0.1, 5.0, 2.0, 0.1, 
                             help="Higher values make watermark stronger but may affect text quality")
    delta = st.sidebar.slider("Detection Threshold (δ)", 0.5, 5.0, 2.0, 0.1,
                             help="Higher values reduce false positives but increase false negatives")
    
    watermarker.gamma = gamma
    watermarker.delta = delta
    
    # Navigation
    st.sidebar.title("📑 Navigation")
    page = st.sidebar.radio("Select Page", ["Text Generation", "Watermark Detection"])
    
    if page == "Text Generation":
        generation_page(watermarker)
    else:
        detection_page(watermarker)

def generation_page(watermarker):
    """
    Page for generating text with or without watermarks.
    """
    st.title("💧 LLM Text Generation with Watermarking")
    st.write("""
    Generate text with or without an invisible watermark that can later be detected.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area("Enter a prompt:", 
                            "The future of artificial intelligence is", 
                            height=100)
    
    with col2:
        use_watermark = st.toggle("Apply Watermark", True)
        max_tokens = st.slider("Max Tokens", 10, 200, 50)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    
    if st.button("Generate Text", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Callback to update progress bar
        def update_progress(step, total, text=None):
            progress_bar.progress(step / total)
            status_text.text(f"Generating: {step}/{total} tokens")
        
        # Generate text
        start_time = time.time()
        generated_text, tokens, green_lists = watermarker.generate_watermarked_text(
            prompt=prompt,
            max_length=max_tokens,
            temperature=temperature,
            use_watermark=use_watermark,
            progress_callback=update_progress
        )
        
        # Display results
        status_text.text(f"Generated {len(tokens) - len(prompt.split())} tokens in {time.time() - start_time:.2f} seconds")
        st.subheader("Generated Text")
        st.write(generated_text)
        
        # Add copy button
        st.text_area("Copy text:", generated_text, height=150)
        
        # Add information about the watermark
        if use_watermark:
            st.success("✅ Text was generated WITH watermark")
            st.info("To detect this watermark, go to the 'Watermark Detection' page and paste this text.")
        else:
            st.warning("⚠️ Text was generated WITHOUT watermark")

def detection_page(watermarker):
    """
    Page for detecting watermarks in text and highlighting tokens.
    """
    st.title("🔍 LLM Watermark Detection")
    st.write("""
    Detect if text contains a watermark and visualize the watermarked tokens.
    Paste text from the generation page or from another source to analyze it.
    """)
    
    text_to_analyze = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Detect Watermark", type="primary", use_container_width=True):
        if not text_to_analyze:
            st.error("Please enter some text to analyze.")
            return
        
        # Perform detection with token colors
        detection_result = watermarker.detect_watermark(
            text_to_analyze, 
            return_token_colors=True
        )
        
        # Create columns for results display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display visual highlighting
            st.subheader("Highlighted Text Analysis")
            
            # Create a container with scrolling for the highlighted text
            html_output = highlight_text(detection_result["token_colors"])
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    height: 300px; 
                    overflow-y: auto;
                    background-color: white;
                    font-size: 16px;
                    line-height: 1.5;
                ">
                {html_output}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Color legend
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 10px;">
                <div style="background-color: rgba(0, 255, 0, 0.3); padding: 3px 8px; margin-right: 10px; border-radius: 3px;">Green tokens</div>
                <div style="background-color: rgba(255, 0, 0, 0.2); padding: 3px 8px; margin-right: 10px; border-radius: 3px;">Red tokens</div>
                <div style="color: gray; padding: 3px 8px; border-radius: 3px;">Neutral (context tokens)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Display detection results
            result_color = "green" if detection_result["has_watermark"] else "red"
            result_icon = "✅" if detection_result["has_watermark"] else "❌"
            result_text = "WATERMARK DETECTED" if detection_result["has_watermark"] else "NO WATERMARK DETECTED"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {'rgba(0, 255, 0, 0.2)' if detection_result['has_watermark'] else 'rgba(255, 0, 0, 0.2)'};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 18px;
                    margin-bottom: 20px;
                ">
                {result_icon} {result_text}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Statistical results
            if "num_tokens_analyzed" in detection_result and detection_result["num_tokens_analyzed"] > 0:
                st.metric("Z-Score", f"{detection_result['z_score']:.2f}", 
                         delta=f"{detection_result['z_score'] - watermarker.delta:.2f} vs threshold")
                
                # Confidence level based on p-value
                confidence = (1 - detection_result["p_value"]) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Token statistics
                st.markdown("### Token Statistics")
                st.write(f"Total tokens analyzed: {detection_result['num_tokens_analyzed']}")
                
                # Green vs Red distribution
                green_count = detection_result.get("green_count", 0)
                red_count = detection_result.get("red_count", 0)
                
                green_pct = green_count / detection_result["num_tokens_analyzed"] * 100
                red_pct = red_count / detection_result["num_tokens_analyzed"] * 100
                
                # Create a horizontal stacked bar chart to show distribution
                st.markdown(
                    f"""
                    <div style="margin-top: 10px;">
                        <div style="display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin-bottom: 5px;">
                            <div style="width: {green_pct}%; background-color: rgba(0, 255, 0, 0.5);"></div>
                            <div style="width: {red_pct}%; background-color: rgba(255, 0, 0, 0.3);"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 12px;">
                            <div>Green: {green_count} ({green_pct:.1f}%)</div>
                            <div>Red: {red_count} ({red_pct:.1f}%)</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Expected distribution
                st.write(f"Expected green: {detection_result['expected_green_fraction']*100:.1f}%")
                
                # Plot
                fig = watermarker.visualize_detection(detection_result)
                st.pyplot(fig)
            else:
                st.warning("Not enough tokens to analyze. Please provide more text.")

if __name__ == "__main__":
    main()
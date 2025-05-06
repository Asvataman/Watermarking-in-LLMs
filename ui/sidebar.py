import streamlit as st
from typing import Dict
from config import (
    WATERMARK_GAMMA,
    WATERMARK_DELTA,
    DETECTION_THRESHOLD,
    WATERMARK_SEED
)

class Sidebar:
    """Sidebar for the Streamlit app with watermarking settings."""
    
    def __init__(self):
        """Initialize the sidebar."""
        pass
    
    def render(self) -> Dict:
        """
        Render the sidebar with controls for watermark parameters.
        
        Returns:
            Dictionary of watermark parameters
        """
        st.sidebar.title("LLM Watermarking")
        st.sidebar.markdown("## About")
        st.sidebar.markdown(
            "This application demonstrates watermarking for large language models, "
            "based on the paper by Kirchenbauer et al. The watermark works by subtly "
            "biasing the model's token selection process, creating a statistical pattern "
            "that can be detected later."
        )
        
        st.sidebar.markdown("## Watermark Parameters")
        
        # Add parameters control
        gamma = st.sidebar.slider(
            "Green List Size (γ)", 
            min_value=0.1, 
            max_value=0.9, 
            value=WATERMARK_GAMMA,
            step=0.05,
            help="Proportion of vocabulary assigned to the green list (0-1)"
        )
        
        delta = st.sidebar.slider(
            "Logit Bias (δ)", 
            min_value=0.1, 
            max_value=10.0, 
            value=WATERMARK_DELTA,
            step=0.5,
            help="Bias added to green list token logits - higher values create a stronger watermark"
        )
        
        threshold = st.sidebar.slider(
            "Detection Threshold (z-score)", 
            min_value=1.0, 
            max_value=10.0, 
            value=DETECTION_THRESHOLD,
            step=0.5,
            help="Z-score threshold for watermark detection"
        )
        
        # Advanced settings in an expander
        with st.sidebar.expander("Advanced Settings"):
            seed = st.number_input(
                "Watermark Seed", 
                min_value=0, 
                max_value=9999999, 
                value=WATERMARK_SEED,
                help="Seed for the watermark hash function"
            )
            
            st.markdown(
                "This implementation uses a simplified approach where only the immediately preceding "
                "token determines the green list for the current token. This matches the basic "
                "approach described in the paper and ensures consistency between generation and detection."
            )
        
        # Add info about the paper
        st.sidebar.markdown("## Reference")
        st.sidebar.markdown(
            "Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). "
            "A Watermark for Large Language Models. arXiv:2301.10226"
        )
        
        # Return parameters
        return {
            "gamma": gamma,
            "delta": delta,
            "threshold": threshold,
            "seed": seed
        }
import streamlit as st
import time

def create_sidebar():
    """Create the sidebar with watermarking settings."""
    st.sidebar.title("LLM Watermarking")
    st.sidebar.markdown("""
    This application demonstrates the watermarking technique described in the paper 
    "A Watermark for Large Language Models" by Kirchenbauer et al. (2023).
    """)
    
    # Main settings section
    st.sidebar.header("Settings")
    
    # Watermarking toggle
    watermarking_enabled = st.sidebar.toggle(
        "Enable Watermarking",
        value=st.session_state.get("watermarking_enabled", True),
        help="Toggle watermarking on or off for generated text"
    )
    
    # Update the session state
    if watermarking_enabled != st.session_state.get("watermarking_enabled", True):
        st.session_state.watermarking_enabled = watermarking_enabled
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        # Watermarking parameters
        gamma = st.slider(
            "Gamma (Green Token Ratio)",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.get("watermark_gamma", 0.5),
            step=0.1,
            help="Fraction of tokens marked as 'green' (watermark tokens)"
        )
        
        delta = st.slider(
            "Delta (Logit Bias)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.get("watermark_delta", 2.0),
            step=0.5,
            help="Logit bias applied to green tokens"
        )
        
        seed = st.number_input(
            "Watermark Seed",
            min_value=1,
            max_value=100000,
            value=st.session_state.get("watermark_seed", 42),
            help="Random seed for the watermark key"
        )
        
        z_threshold = st.slider(
            "Z-threshold (Detection)",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.get("z_threshold", 4.0),
            step=0.5,
            help="Z-score threshold for watermark detection"
        )
        
        # Update the session state
        if gamma != st.session_state.get("watermark_gamma", 0.5):
            st.session_state.watermark_gamma = gamma
            
        if delta != st.session_state.get("watermark_delta", 2.0):
            st.session_state.watermark_delta = delta
            
        if seed != st.session_state.get("watermark_seed", 42):
            st.session_state.watermark_seed = seed
            
        if z_threshold != st.session_state.get("z_threshold", 4.0):
            st.session_state.z_threshold = z_threshold
    
    # Information section
    with st.sidebar.expander("About Watermarking"):
        st.markdown("""
        ### How It Works
        
        **Watermarking** works by subtly biasing the LLM's generation process:
        
        1. Using a secret key, tokens are divided into "green" and "red" lists
        2. The model is biased to slightly prefer green tokens during generation
        3. This creates a statistical pattern detectable with the right key
        
        **Detection** analyzes the text to find if green tokens appear more frequently 
        than would be expected by chance.
        
        ### Parameters
        
        - **Gamma**: Controls the fraction of tokens marked as "green"
        - **Delta**: Controls the strength of the bias toward green tokens
        - **Z-threshold**: Controls detection sensitivity (higher = fewer false positives)
        
        ### Use Cases
        
        - Verify content authenticity
        - Detect AI-generated misinformation
        - Attribution for AI-generated content
        - Academic integrity
        """)
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        
        # Clear any analysis states
        for key in list(st.session_state.keys()):
            if key.startswith("analysis_"):
                del st.session_state[key]
                
        st.rerun()


def initialize_sidebar_state():
    """Initialize session state variables for the sidebar."""
    if "watermarking_enabled" not in st.session_state:
        st.session_state.watermarking_enabled = True
        
    if "watermark_gamma" not in st.session_state:
        st.session_state.watermark_gamma = 0.5
        
    if "watermark_delta" not in st.session_state:
        st.session_state.watermark_delta = 2.0
        
    if "watermark_seed" not in st.session_state:
        st.session_state.watermark_seed = 42
        
    if "z_threshold" not in st.session_state:
        st.session_state.z_threshold = 4.0
import streamlit as st
from typing import List, Dict, Tuple, Optional
import time
from watermarking import Watermarker, WatermarkDetector
from utils import get_azure_client, preprocess_text
from config import (
    AZURE_OPENAI_DEPLOYMENT_NAME,
    WATERMARK_GAMMA,
    WATERMARK_DELTA,
    WATERMARK_SEED,
    DETECTION_THRESHOLD
)

class ChatInterface:
    """Streamlit chat interface for interacting with the watermarked LLM."""
    
    def __init__(self):
        """Initialize the chat interface."""
        # Initialize chat history in session state if not already present
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "watermark_enabled" not in st.session_state:
            st.session_state.watermark_enabled = False
            
        if "detection_results" not in st.session_state:
            st.session_state.detection_results = {}
            
        # Initialize OpenAI client
        self.client = get_azure_client()
        
        # Initialize watermarker and detector
        self.watermarker = Watermarker(
            gamma=WATERMARK_GAMMA,
            delta=WATERMARK_DELTA,
            seed=WATERMARK_SEED
        )
        
        self.detector = WatermarkDetector(
            gamma=WATERMARK_GAMMA,
            seed=WATERMARK_SEED,
            threshold=DETECTION_THRESHOLD
        )
    
    def render(self):
        """Render the chat interface."""
        # Display chat header
        st.markdown("### Azure GPT-4o Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Add detection panel for assistant messages
                if message["role"] == "assistant" and message.get("detection_result"):
                    self._render_detection_panel(message["detection_result"], message["content"])
        
        # Chat input with watermark toggle
        st.markdown("---")
        cols = st.columns([8, 2])
        
        with cols[1]:
            watermark_enabled = st.toggle("Enable Watermark", value=st.session_state.watermark_enabled)
            if watermark_enabled != st.session_state.watermark_enabled:
                st.session_state.watermark_enabled = watermark_enabled
                
            if watermark_enabled:
                st.caption("Using soft watermarking (γ={}, δ={})".format(WATERMARK_GAMMA, WATERMARK_DELTA))
                # Remove tooltip call that was causing the error
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    response = self._generate_response(prompt)
                
                # Display response
                message_placeholder.markdown(response["content"])
                
                # Add detection panel
                if response.get("detection_result"):
                    self._render_detection_panel(response["detection_result"], response["content"])
                
                # Add assistant message to chat history
                st.session_state.messages.append(response)
    
    def _generate_response(self, prompt: str) -> Dict:
        """
        Generate a response from the model with optional watermarking.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Dictionary with response content and detection result
        """
        # Prepare messages for the API
        messages = []
        
        # Add previous messages for context
        for message in st.session_state.messages:
            messages.append({"role": message["role"], "content": message["content"]})
        
        # Generate response
        start_time = time.time()
        response_text = ""
        watermark_status = "disabled"
        
        try:
            if st.session_state.watermark_enabled:
                # Try to generate with watermarking
                try:
                    with st.spinner("Generating response with watermark..."):
                        response = self.watermarker.generate_with_watermark(
                            client=self.client,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=1000,
                            stream=False,
                            model=AZURE_OPENAI_DEPLOYMENT_NAME
                        )
                        response_text = response.choices[0].message.content
                        watermark_status = "enabled"
                except Exception as e:
                    st.error(f"Error with watermarking: {str(e)}")
                    st.info("Falling back to generation without watermark")
                    
                    # Fall back to non-watermarked generation
                    response = self.client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000,
                        stream=False
                    )
                    response_text = response.choices[0].message.content
                    watermark_status = "failed"
            else:
                # Generate without watermarking
                response = self.client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                    stream=False
                )
                response_text = response.choices[0].message.content
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            response_text = f"I'm sorry, but there was an error generating a response. Please try again later.\n\nError details: {str(e)}"
            watermark_status = "error"
        
        # Detect watermark in the response
        detection_result = self.detector.detect(response_text)
        
        generation_time = time.time() - start_time
        
        return {
            "role": "assistant",
            "content": response_text,
            "detection_result": {
                **detection_result,
                "watermark_enabled": st.session_state.watermark_enabled,
                "watermark_status": watermark_status,
                "generation_time": generation_time
            }
        }
    
    def _render_detection_panel(self, detection_result: Dict, content: str):
        """
        Render the watermark detection panel.
        
        Args:
            detection_result: Detection result dictionary
            content: Original text content
        """
        # Import scipy.stats here to avoid NameError
        from scipy import stats
        
        # Create expander for detection details
        with st.expander("Inspect Response (Watermark Detection)"):
            cols = st.columns(2)
            
            with cols[0]:
                st.markdown("#### Watermark Detection")
                
                # Show watermark status
                if detection_result.get('watermark_status') == "enabled":
                    st.success("✅ Watermarking: Enabled and working")
                elif detection_result.get('watermark_status') == "failed":
                    st.warning("⚠️ Watermarking: Attempted but failed, fell back to normal generation")
                elif detection_result.get('watermark_status') == "error":
                    st.error("❌ Watermarking: Error in generation")
                else:
                    st.info("ℹ️ Watermarking: Disabled")
                
                # Detection results
                st.markdown(f"**Is Watermarked:** {'Yes' if detection_result['is_watermarked'] else 'No'}")
                st.markdown(f"**Z-Score:** {detection_result['z_score']:.2f}")
                st.markdown(f"**P-Value:** {detection_result['p_value']:.8f}")
                
                # Only calculate percentage if total_tokens > 0
                if detection_result['total_tokens'] > 0:
                    percentage = detection_result['green_token_count'] / detection_result['total_tokens'] * 100
                    st.markdown(f"**Green Tokens:** {detection_result['green_token_count']} / {detection_result['total_tokens']} "
                               f"({percentage:.1f}%)")
                else:
                    st.markdown(f"**Green Tokens:** {detection_result['green_token_count']} / {detection_result['total_tokens']}")
                
                st.markdown(f"**Expected Green Tokens:** {detection_result['expected_green_count']:.1f} "
                           f"({WATERMARK_GAMMA * 100:.1f}%)")
                st.markdown(f"**Generation Time:** {detection_result['generation_time']:.2f} seconds")
                
                # Threshold explanation with proper import
                st.markdown(f"**Detection Threshold:** z-score > {DETECTION_THRESHOLD} (p < {1 - stats.norm.cdf(DETECTION_THRESHOLD):.8f})")
            
            with cols[1]:
                if detection_result["visualization"]:
                    st.image(f"data:image/png;base64,{detection_result['visualization']}")
                else:
                    st.info("Visualization not available (text too short for reliable detection)")
            
            # Display interpretation
            st.markdown("#### Interpretation")
            if detection_result['is_watermarked']:
                st.success(
                    "📝 This text shows a statistically significant watermark pattern (z-score > {:.1f}). "
                    "There is strong evidence that this was generated with watermarking enabled."
                    .format(DETECTION_THRESHOLD)
                )
            elif detection_result['z_score'] > 2.0:
                st.info(
                    "🔍 This text shows some pattern (z-score: {:.2f}), but it's below our detection threshold ({:.1f}). "
                    "This might indicate watermarking that has been modified or diluted."
                    .format(detection_result['z_score'], DETECTION_THRESHOLD)
                )
            else:
                if detection_result.get('watermark_status') == "enabled":
                    st.info(
                        "🔍 No significant watermark pattern detected despite watermarking being enabled. "
                        "This could be due to an API limitation that caused fallback to non-watermarked generation, "
                        "or because the text contains mostly low-entropy tokens that are less influenced by watermarking."
                    )
                else:
                    st.info(
                        "🔍 No significant watermark pattern detected. This text was likely generated without watermarking."
                    )
            
            # Display highlighted text
            st.markdown("#### Highlighted Response (Green Tokens)")
            highlighted_text = self.detector.get_highlighted_text(content, detection_result)
            st.markdown(highlighted_text, unsafe_allow_html=True)
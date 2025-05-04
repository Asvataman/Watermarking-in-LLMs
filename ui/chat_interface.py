import streamlit as st
import time
from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio

def initialize_chat_state():
    """Initialize the chat-related session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    if "watermarking_enabled" not in st.session_state:
        st.session_state.watermarking_enabled = True


def display_chat_messages():
    """Display all messages in the chat history with watermark analysis buttons."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add "Check Watermark" button for assistant messages
            if message["role"] == "assistant":
                button_key = f"check_watermark_{i}"
                
                # Create a unique key for the analysis state
                analysis_key = f"analysis_{i}"
                if analysis_key not in st.session_state:
                    st.session_state[analysis_key] = None
                
                # Check if analysis already exists
                if st.session_state[analysis_key] is not None:
                    # Display the analysis result
                    with st.sidebar:
                        display_watermark_analysis(message["content"], st.session_state[analysis_key])
                
                # Add the button
                if st.button("Check Watermark", key=button_key):
                    # Set the analysis state
                    from watermarking.detector import WatermarkDetector
                    detector = WatermarkDetector(
                        seed=st.session_state.get("watermark_seed", 42),
                        gamma=st.session_state.get("watermark_gamma", 0.5)
                    )
                    analysis = detector.analyze_text_segments(message["content"])
                    st.session_state[analysis_key] = analysis
                    
                    # Force a rerun to display the analysis
                    st.rerun()


async def process_user_input(
    user_input: str,
    get_response_func: Callable[[List[Dict[str, str]]], Awaitable[str]]
):
    """
    Process user input, get AI response, and update chat history.
    
    Args:
        user_input: User's input text
        get_response_func: Async function to get AI response
    """
    if not user_input or st.session_state.processing:
        return
    
    # Set processing flag
    st.session_state.processing = True
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    try:
        response = await get_response_func(st.session_state.messages)
        
        # Add AI response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
    finally:
        # Reset processing flag
        st.session_state.processing = False


def create_chat_input(on_submit_func):
    """
    Create the chat input component.
    
    Args:
        on_submit_func: Function to call when input is submitted
    """
    # Create the input box
    user_input = st.chat_input(
        "Ask something...",
        disabled=st.session_state.processing
    )
    
    # Process input if provided
    if user_input:
        asyncio.run(on_submit_func(user_input))


def display_watermark_analysis(text: str, analysis: Dict):
    """
    Display watermark analysis results in the sidebar.
    
    Args:
        text: Text that was analyzed
        analysis: Analysis results from the detector
    """
    st.sidebar.header("Watermark Analysis")
    
    # Display overall result
    st.sidebar.subheader("Overall Analysis")
    overall = analysis["overall"]
    
    # Determine the result color
    if overall["is_watermarked"]:
        result_color = "green"
        result_text = "WATERMARK DETECTED ✓"
    else:
        result_color = "red"
        result_text = "NO WATERMARK DETECTED ✗"
    
    # Display the result with colored box
    st.sidebar.markdown(
        f"""
        <div style="padding: 10px; background-color: {'rgba(0, 255, 0, 0.1)' if overall['is_watermarked'] else 'rgba(255, 0, 0, 0.1)'}; 
                    border-radius: 5px; border: 1px solid {result_color};">
            <h3 style="color: {result_color}; margin: 0;">{result_text}</h3>
            <p>Confidence: {overall['confidence']:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display statistics
    st.sidebar.markdown("""
    ### Statistical Analysis
    """)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Z-score", f"{overall['z_score']:.2f}")
    col2.metric("p-value", f"{overall['p_value']:.4f}")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Green tokens", overall["green_count"])
    col2.metric("Red tokens", overall["red_count"])
    
    st.sidebar.metric("Total tokens analyzed", overall["analyzed_pairs"])
    
    # Show detailed segment analysis
    if len(analysis["segments"]) > 1:
        st.sidebar.markdown("### Segment Analysis")
        
        # Create a chart of segment confidence scores
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(8, 3))
        segments = analysis["segments"]
        x = np.arange(len(segments))
        confidences = [segment["confidence"] for segment in segments]
        
        # Create the bar chart
        bars = ax.bar(x, confidences, color=['green' if c > 50 else 'red' for c in confidences])
        ax.set_xlabel('Text Segment')
        ax.set_ylabel('Confidence %')
        ax.set_title('Watermark Confidence by Text Segment')
        ax.set_ylim(0, 100)
        
        # Add a horizontal line at the detection threshold
        threshold = 50
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
        
        st.sidebar.pyplot(fig)
        
        # Show the most confident segment
        if any(segment["is_watermarked"] for segment in segments):
            st.sidebar.markdown("### Most Confident Watermarked Segment")
            
            # Find the segment with the highest confidence that is watermarked
            watermarked_segments = [s for s in segments if s["is_watermarked"]]
            if watermarked_segments:
                most_confident = max(watermarked_segments, key=lambda s: s["confidence"])
                
                st.sidebar.markdown(f"""
                **Confidence:** {most_confident['confidence']:.2f}%  
                **Z-score:** {most_confident['z_score']:.2f}  
                **Text excerpt:**
                """)
                
                # Truncate if too long
                excerpt = most_confident["text"]
                if len(excerpt) > 300:
                    excerpt = excerpt[:300] + "..."
                
                st.sidebar.markdown(f"> {excerpt}")
    
    # Add a close button
    if st.sidebar.button("Close Analysis"):
        st.session_state[f"analysis_{len(st.session_state.messages) - 1}"] = None
        st.rerun()
import streamlit as st
from ui import ChatInterface, Sidebar, DetectionPanel
from watermarking import Watermarker, WatermarkDetector
from config import (
    WATERMARK_GAMMA,
    WATERMARK_DELTA,
    WATERMARK_SEED,
    DETECTION_THRESHOLD
)

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="GPT-4o Watermarking Demo",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the sidebar
    sidebar = Sidebar()
    params = sidebar.render()
    
    # Create watermark detector with parameters from UI
    detector = WatermarkDetector(
        gamma=params["gamma"],
        seed=params["seed"],
        threshold=params["threshold"]
    )
    
    # App pages/tabs
    tab1, tab2 = st.tabs(["Chat", "Watermark Detector"])
    
    with tab1:
        # Initialize and render chat interface
        chat_interface = ChatInterface()
        chat_interface.render()
    
    with tab2:
        # Initialize and render detection panel
        detection_panel = DetectionPanel(detector)
        detection_panel.render()
    
if __name__ == "__main__":
    main()
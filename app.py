import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from watermarking.watermarker import Watermarker, AzureWatermarkedClient
from watermarking.detector import WatermarkDetector
from utils.azure_client import AzureOpenAIClient
from ui.chat_interface import initialize_chat_state, display_chat_messages, process_user_input, create_chat_input
from ui.sidebar import initialize_sidebar_state, create_sidebar
from ui.detection_panel import create_detection_panel

# Set page config
st.set_page_config(
    page_title="LLM Watermarking Demo",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_chat_state()
initialize_sidebar_state()

# Main application
def main():
    # Display header
    st.title("🌊 LLM Watermarking")
    
    # Create tabs for chat and detector
    chat_tab, detector_tab = st.tabs(["Chat Interface", "Standalone Detector"])
    
    with chat_tab:
        st.markdown("""
        Chat with GPT-4o with toggleable watermarking. You can check any response for watermarks 
        using the "Check Watermark" button that appears with each assistant message.
        """)
        
        # Initialize watermarked client (lazily to avoid errors on startup)
        if "azure_client" not in st.session_state:
            try:
                client = AzureOpenAIClient()
                watermarker = Watermarker(
                    gamma=st.session_state.watermark_gamma,
                    delta=st.session_state.watermark_delta,
                    seed=st.session_state.watermark_seed
                )
                st.session_state.azure_client = client
                st.session_state.watermarker = watermarker
                st.session_state.watermarked_client = AzureWatermarkedClient(client, watermarker)
            except Exception as e:
                st.error(f"Error initializing Azure client: {str(e)}")
                st.info("Please check your Azure OpenAI credentials in the .env file")
                
                # Create dummy placeholders
                st.session_state.azure_client = None
                st.session_state.watermarker = None
                st.session_state.watermarked_client = None
                
        # Display watermarking status
        watermarking_status = "ON ✅" if st.session_state.watermarking_enabled else "OFF ❌"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <p style="margin: 0; margin-right: 10px;">Watermarking:</p>
                <p style="margin: 0; font-weight: bold; color: {'green' if st.session_state.watermarking_enabled else 'red'};">
                    {watermarking_status}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display chat messages
        display_chat_messages()
        
        # Create chat input
        if st.session_state.azure_client is not None:
            async def get_response(messages):
                """Get response from Azure OpenAI with watermarking."""
                # Toggle watermarking based on session state
                st.session_state.watermarked_client.toggle_watermarking(st.session_state.watermarking_enabled)
                
                # Update watermarker parameters if they've changed
                st.session_state.watermarker.gamma = st.session_state.watermark_gamma
                st.session_state.watermarker.delta = st.session_state.watermark_delta
                st.session_state.watermarker.seed = st.session_state.watermark_seed
                
                # Format messages for API
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Get response
                response = await st.session_state.watermarked_client.generate_completion(
                    formatted_messages,
                    temperature=0.7,
                    max_tokens=800
                )
                print(response)
                
                return response.choices[0].message.content
            
            # Create the chat input
            create_chat_input(lambda user_input: process_user_input(user_input, get_response))
            display_chat_messages()
        else:
            st.warning("Please configure your Azure OpenAI credentials to use the chat interface.")
    
    with detector_tab:
        create_detection_panel()
    
    # Create sidebar
    create_sidebar()

# Run the app
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def create_detection_panel():
    """
    Create a detection panel for analyzing arbitrary text.
    
    This panel allows users to input text and analyze it for watermarks
    independently of the chat interface.
    """
    st.header("Watermark Detector")
    st.markdown("""
    Paste text below to analyze it for the presence of the watermark.
    """)
    
    # Text input area
    text_to_analyze = st.text_area(
        "Text to analyze",
        height=200,
        help="Paste text here to check for watermarks"
    )
    
    # Analysis parameters
    col1, col2 = st.columns(2)
    
    with col1:
        seed = st.number_input(
            "Watermark Seed",
            min_value=1,
            max_value=100000,
            value=st.session_state.get("watermark_seed", 42),
            help="Secret key seed for watermark detection (must match the seed used for watermarking)"
        )
    
    with col2:
        gamma = st.slider(
            "Gamma",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.get("watermark_gamma", 0.5),
            step=0.1,
            help="Fraction of tokens expected to be 'green'"
        )
    
    # Analyze button
    analyze_button = st.button("Analyze Text")
    
    # Perform analysis when button is clicked
    if analyze_button and text_to_analyze:
        from watermarking.detector import WatermarkDetector
        
        # Create detector with the specified parameters
        detector = WatermarkDetector(
            seed=seed,
            gamma=gamma,
            z_threshold=st.session_state.get("z_threshold", 4.0)
        )
        
        # Analyze the text
        with st.spinner("Analyzing text..."):
            analysis = detector.analyze_text_segments(text_to_analyze)
            
        # Display results
        display_analysis_results(analysis)


def display_analysis_results(analysis: Dict):
    """
    Display the watermark analysis results.
    
    Args:
        analysis: Analysis results from the detector
    """
    st.subheader("Analysis Results")
    
    # Overall result
    overall = analysis["overall"]
    
    # Determine result color and text
    if overall["is_watermarked"]:
        result_color = "green"
        result_text = "WATERMARK DETECTED"
        emoji = "✅"
    else:
        result_color = "red"
        result_text = "NO WATERMARK DETECTED"
        emoji = "❌"
    
    # Display the result with colored box
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: {'rgba(0, 255, 0, 0.1)' if overall['is_watermarked'] else 'rgba(255, 0, 0, 0.1)'}; 
                    border-radius: 10px; border: 2px solid {result_color}; margin-bottom: 20px;">
            <h2 style="color: {result_color}; margin: 0;">{result_text} {emoji}</h2>
            <h3>Confidence: {overall['confidence']:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Z-score", f"{overall['z_score']:.3f}")
    
    with col2:
        st.metric("p-value", f"{overall['p_value']:.6f}")
    
    with col3:
        st.metric("Total Tokens", overall['total_tokens'])
    
    # Green vs Red tokens
    st.subheader("Token Distribution")
    
    # Create a donut chart for green vs red tokens
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data for pie chart
    green = overall["green_count"]
    red = overall["red_count"]
    
    # Create pie chart
    ax.pie(
        [green, red], 
        labels=['Green Tokens', 'Red Tokens'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['green', 'red'],
        wedgeprops=dict(width=0.5)
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Display the chart
    st.pyplot(fig)
    
    # Display segment analysis if multiple segments exist
    if len(analysis["segments"]) > 1:
        st.subheader("Segment Analysis")
        
        # Create a dataframe for the segments
        segment_data = []
        for i, segment in enumerate(analysis["segments"]):
            segment_data.append({
                "Segment": i + 1,
                "Start Token": segment["start_token"],
                "End Token": segment["end_token"],
                "Watermarked": "Yes" if segment["is_watermarked"] else "No",
                "Confidence (%)": f"{segment['confidence']:.2f}",
                "Z-score": f"{segment['z_score']:.2f}"
            })
        
        # Display as a table
        df = pd.DataFrame(segment_data)
        st.dataframe(df, hide_index=True)
        
        # Create a chart of segment confidence scores
        fig, ax = plt.subplots(figsize=(10, 4))
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
        
        st.pyplot(fig)
        
        # Show the most confident segment
        if any(segment["is_watermarked"] for segment in segments):
            st.subheader("Most Confident Watermarked Segment")
            
            # Find the segment with the highest confidence that is watermarked
            watermarked_segments = [s for s in segments if s["is_watermarked"]]
            if watermarked_segments:
                most_confident = max(watermarked_segments, key=lambda s: s["confidence"])
                
                st.markdown(f"""
                **Confidence:** {most_confident['confidence']:.2f}%  
                **Z-score:** {most_confident['z_score']:.2f}  
                """)
                
                with st.expander("View Segment Text"):
                    st.markdown(most_confident["text"])

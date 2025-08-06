#!/usr/bin/env python3
"""
Streamlit Squat Detection and Form Analysis Application

This Streamlit app provides a web interface for real-time squat detection and form analysis
using MediaPipe pose estimation and machine learning models.

Usage:
    streamlit run streamlit_app.py

Features:
    - Real-time webcam squat detection
    - Video file upload and analysis
    - Live form feedback and rep counting
    - Audio feedback for form corrections
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import time
import warnings
from PIL import Image
import mediapipe as mp
from squat_module import SquatDetector

warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils import rescale_frame

# Import configuration for MediaPipe settings
try:
    from config import (
        MEDIAPIPE_DETECTION_CONFIDENCE,
        MEDIAPIPE_TRACKING_CONFIDENCE
    )
except ImportError:
    # Fallback values if config.py is not available
    MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Page configuration
st.set_page_config(
    page_title="Squat Form Analyzer",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        detector = SquatDetector(model_path="./model/LR_model.pkl")
        return detector
    except Exception as e:
        st.error(f"Error loading squat detection model: {e}")
        return None

def process_video_frame(frame, pose, detector, prediction_threshold=0.6):
    """Process a single frame for squat analysis"""
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the frame
    results = pose.process(image)
    
    # Convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not results.pose_landmarks:
        cv2.putText(image, "NO POSE DETECTED", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image, None, None, None
    
    try:
        # Process frame with detector
        processed_frame = detector.process_frame(image, pose)
        
        # Get current analysis results
        if results.pose_landmarks:
            predicted_stage, confidence = detector.predict_squat_stage(results)
            
            # Analyze form
            analysis_stage = detector.current_stage if detector.current_stage else predicted_stage
            analysis_results = detector.analyze_foot_knee_placement(results, analysis_stage)
            
            return processed_frame, predicted_stage, confidence, analysis_results
        else:
            return image, None, None, None
            
    except Exception as e:
        cv2.putText(image, f"ERROR: {str(e)[:30]}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return image, None, None, None

def main():
    """Main Streamlit application."""
    st.title("🏋️‍♂️ Squat Form Analyzer")
    st.markdown("Upload a video or use your webcam to analyze squat form using AI")
    
    # Load model
    detector = load_model()
    if detector is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Settings")
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["./model/LR_model.pkl", "./model/sklearn_models.pkl"],
        index=0
    )
    
    # Detection settings
    st.sidebar.subheader("⚙️ Detection Settings")
    prediction_threshold = st.sidebar.slider("Prediction Confidence Threshold", 0.5, 1.0, 0.6, 0.05)
    resize_percent = st.sidebar.slider("Frame Resize (%)", 25, 100, 75, 5)
    
    # Mode selection
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Upload Video", "Live Webcam"]
    )
    
    # Session state for webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'squat_detector' not in st.session_state:
        st.session_state.squat_detector = None
    
    if analysis_mode == "Upload Video":
        st.header("📹 Video Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Analysis Results")
                video_placeholder = st.empty()
                
            with col2:
                st.subheader("Statistics")
                reps_placeholder = st.empty()
                stage_placeholder = st.empty()
                confidence_placeholder = st.empty()
                form_placeholder = st.empty()
                
            # Process video
            if st.button("🚀 Analyze Video", type="primary"):
                cap = cv2.VideoCapture(tfile.name)
                video_detector = SquatDetector(model_path=model_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Progress bar
                progress_bar = st.progress(0)
                frame_count = 0
                
                with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize frame
                        frame = rescale_frame(frame, resize_percent)
                        
                        # Process frame
                        processed_frame, predicted_stage, confidence, analysis_results = process_video_frame(
                            frame, pose, video_detector, prediction_threshold
                        )
                        
                        # Update display
                        video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                        
                        # Update statistics
                        reps_placeholder.metric("Total Reps", f"{video_detector.counter:.1f}")
                        
                        if predicted_stage:
                            stage_placeholder.metric("Current Stage", predicted_stage.title())
                            confidence_placeholder.metric("Confidence", f"{confidence:.2f}")
                            
                            if analysis_results:
                                # Form feedback
                                foot_status = "Correct" if analysis_results.get("foot_placement", -1) == 0 else "Incorrect"
                                knee_status = "Correct" if analysis_results.get("knee_placement_up", -1) == 0 else "Incorrect"
                                form_placeholder.text(f"Foot: {foot_status}\nKnee: {knee_status}")
                        else:
                            stage_placeholder.metric("Current Stage", "Not Detected")
                            confidence_placeholder.metric("Confidence", "0.00")
                            form_placeholder.text("Form: Not Detected")
                        
                        # Update progress
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                        
                        # Small delay to make video viewable
                        time.sleep(1 / fps if fps > 0 else 0.033)
                
                cap.release()
                os.unlink(tfile.name)  # Delete temporary file
                
                st.success(f"Analysis complete! Total reps detected: {video_detector.counter:.1f}")
    
    elif analysis_mode == "Live Webcam":
        st.header("📸 Live Webcam Analysis")
        
        # Webcam controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_button = st.button("Start Analysis", type="primary")
        with col2:
            stop_button = st.button("Stop Analysis")
        with col3:
            reset_button = st.button("Reset Counter")
        
        # Create placeholders for live feed
        video_placeholder = st.empty()
        
        # Statistics placeholders
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            reps_metric = st.empty()
        with col2:
            stage_metric = st.empty()
        with col3:
            confidence_metric = st.empty()
        with col4:
            form_metric = st.empty()
        
        if start_button:
            st.session_state.webcam_active = True
            st.session_state.squat_detector = SquatDetector(model_path=model_path)
        
        if stop_button:
            st.session_state.webcam_active = False
        
        if reset_button and st.session_state.squat_detector:
            st.session_state.squat_detector.reset_counter()
        
        if st.session_state.webcam_active:
            try:
                cap = cv2.VideoCapture(0)
                
                with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                    stframe = st.empty()
                    
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break
                        
                        # Flip frame horizontally for selfie view
                        frame = cv2.flip(frame, 1)
                        
                        # Resize frame
                        frame = rescale_frame(frame, resize_percent)
                        
                        # Process frame
                        processed_frame, predicted_stage, confidence, analysis_results = process_video_frame(
                            frame, pose, st.session_state.squat_detector, prediction_threshold
                        )
                        
                        # Display frame
                        stframe.image(processed_frame, channels="BGR", use_container_width=True)
                        
                        # Update metrics
                        reps_metric.metric("Reps", f"{st.session_state.squat_detector.counter:.1f}")
                        
                        if predicted_stage:
                            stage_metric.metric("Stage", predicted_stage.title())
                            confidence_metric.metric("Confidence", f"{confidence:.2f}")
                            
                            if analysis_results:
                                # Form feedback
                                foot_status = "✅" if analysis_results.get("foot_placement", -1) == 0 else "❌"
                                knee_status = "✅" if analysis_results.get("knee_placement_up", -1) == 0 else "❌"
                                form_metric.metric("Form", f"Foot: {foot_status} Knee: {knee_status}")
                        else:
                            stage_metric.metric("Stage", "Not Detected")
                            confidence_metric.metric("Confidence", "0.00")
                            form_metric.metric("Form", "Not Detected")
                        
                        # Check if stop was pressed
                        if not st.session_state.webcam_active:
                            break
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error accessing webcam: {e}")
                st.session_state.webcam_active = False
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Squat Form Analysis using AI**
        
        This application uses:
        - **MediaPipe** for pose detection and landmark extraction
        - **Machine Learning** model trained on squat movement patterns
        - **Real-time analysis** of form and rep counting
        - **0.5 point scoring system** for granular feedback
        
        **Form Classifications:**
        - ✅ **Correct Form**: Proper squat technique
        - ❌ **Foot Issues**: Stance too narrow or wide
        - ❌ **Knee Issues**: Knees caving in or tracking incorrectly
        
        **Scoring System:**
        - 🔄 **0.5 points** for correct DOWN stage form
        - 🔄 **0.5 points** for correct UP stage form
        - 🏆 **1.0 point** = perfect squat rep
        
        **Tips for best results:**
        - Ensure good lighting
        - Position yourself fully in frame
        - Wear contrasting colors
        - Maintain steady camera position
        - Stand facing the camera
        """)
    
    # Footer information
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'AI Squat Form Analyzer - Powered by MediaPipe and Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

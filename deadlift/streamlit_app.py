import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import tempfile
import os
import time
from audio import DeadliftAudioFeedback
from deadlift_module import DeadliftPoseAnalysis
# Configure page
st.set_page_config(
    page_title="Deadlift Form Analyzer",
    page_icon="🏋️",
    layout="wide"
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



# Load models
@st.cache_resource
def load_models():
    try:
        with open('./model/deadlift.pkl', 'rb') as f:
            deadlift_model = pickle.load(f)
        
        # The model is already a pipeline with StandardScaler included
        # Extract the scaler from the pipeline for reference (optional)
        try:
            scaler = deadlift_model.named_steps['standardscaler']
        except:
            scaler = None
            
        return deadlift_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found in ./model/ directory! {e}")
        return None, None

def process_video_frame(frame, pose, deadlift_analysis, model, scaler, audio_feedback=None, prediction_threshold=0.7):
    """Process a single frame for deadlift analysis"""
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
        return image, None, None
    
    # Draw landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
    )
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get model prediction
        prediction, confidence = deadlift_analysis.analyze_pose(landmarks, model, scaler)
        
        if prediction and confidence > prediction_threshold:
            # Play audio feedback if enabled - only for incorrect form
            if audio_feedback:
                audio_feedback.check_and_play_feedback(prediction, confidence)
            
            # Status box
            cv2.rectangle(image, (0, 0), (600, 80), (245, 117, 16), -1)
            
            # Display rep counter
            cv2.putText(image, "REPS", (15, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(deadlift_analysis.counter) if deadlift_analysis.is_visible else "UNK", 
                       (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display form prediction
            cv2.putText(image, "FORM", (100, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            form_text = prediction.replace('d_', '').replace('_', ' ').upper()
            cv2.putText(image, form_text[:15], (95, 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, form_text[15:] if len(form_text) > 15 else "", (95, 70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display confidence
            cv2.putText(image, "CONF", (350, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"{confidence:.2f}", (345, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display stage
            cv2.putText(image, "STAGE", (450, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, deadlift_analysis.stage.upper(), (445, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Color coding for form feedback
            if prediction == 'd_correct_up' or prediction == 'd_correct_down':
                form_color = (0, 255, 0)  # Green for correct
            else:
                form_color = (0, 0, 255)  # Red for errors
            
            # Draw form status indicator
            cv2.circle(image, (550, 40), 20, form_color, -1)
            
            return image, prediction, confidence
        else:
            # Low confidence prediction
            cv2.rectangle(image, (0, 0), (400, 40), (245, 117, 16), -1)
            cv2.putText(image, "LOW CONFIDENCE PREDICTION", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return image, None, None
    
    except Exception as e:
        cv2.putText(image, f"ERROR: {str(e)[:30]}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return image, None, None

def main():
    st.title("🏋️ Deadlift Form Analyzer")
    st.markdown("Upload a video or use your webcam to analyze deadlift form using AI")
    
    # Load model and scaler
    model_data = load_models()
    if model_data[0] is None:
        st.stop()
    
    model, scaler = model_data
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Audio settings
    st.sidebar.subheader("🔊 Audio Feedback")
    audio_enabled = st.sidebar.checkbox("Enable Audio Feedback", value=True)
    
    if audio_enabled:
        st.sidebar.write("📢 Audio will play for:")
        st.sidebar.write("• Form errors only")
        st.sidebar.write("• Grip corrections")
        st.sidebar.write("• Spine position errors")
        st.sidebar.write("• Silent for correct form")
    
    # Other settings
    st.sidebar.subheader("⚙️ Detection Settings")
    prediction_threshold = st.sidebar.slider("Prediction Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
    visibility_threshold = st.sidebar.slider("Pose Visibility Threshold", 0.5, 1.0, 0.65, 0.05)
    
    # Mode selection
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Upload Video", "Live Webcam"]
    )
    
    # Initialize audio feedback
    audio_feedback = None
    if audio_enabled:
        audio_feedback = DeadliftAudioFeedback()
    
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
                form_placeholder = st.empty()
                confidence_placeholder = st.empty()
                
            # Process video
            cap = cv2.VideoCapture(tfile.name)
            deadlift_analysis = DeadliftPoseAnalysis(visibility_threshold=visibility_threshold)
            
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
                    
                    # Process frame
                    processed_frame, prediction, confidence = process_video_frame(
                        frame, pose, deadlift_analysis, model, scaler, audio_feedback, prediction_threshold
                    )
                    
                    # Update display
                    video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    # Update statistics
                    reps_placeholder.metric("Total Reps", deadlift_analysis.counter)
                    
                    if prediction:
                        form_text = prediction.replace('d_', '').replace('_', ' ').title()
                        form_placeholder.metric("Current Form", form_text)
                        confidence_placeholder.metric("Confidence", f"{confidence:.2f}")
                    else:
                        form_placeholder.metric("Current Form", "Not Detected")
                        confidence_placeholder.metric("Confidence", "0.00")
                    
                    # Update progress
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                    
                    # Small delay to make video viewable
                    time.sleep(1 / fps if fps > 0 else 0.033)
            
            cap.release()
            os.unlink(tfile.name)  # Delete temporary file
            
            st.success(f"Analysis complete! Total reps detected: {deadlift_analysis.counter}")
    
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
        col1, col2, col3 = st.columns(3)
        with col1:
            reps_metric = st.empty()
        with col2:
            form_metric = st.empty()
        with col3:
            confidence_metric = st.empty()
        
        # Session state for webcam
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'deadlift_analysis' not in st.session_state:
            st.session_state.deadlift_analysis = None
        
        if start_button:
            st.session_state.webcam_active = True
            st.session_state.deadlift_analysis = DeadliftPoseAnalysis(visibility_threshold=visibility_threshold)
        
        if stop_button:
            st.session_state.webcam_active = False
        
        if reset_button and st.session_state.deadlift_analysis:
            st.session_state.deadlift_analysis.counter = 0
        
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
                        
                        # Process frame
                        processed_frame, prediction, confidence = process_video_frame(
                            frame, pose, st.session_state.deadlift_analysis, model, scaler, audio_feedback, prediction_threshold
                        )
                        
                        # Display frame
                        stframe.image(processed_frame, channels="BGR", use_container_width=True)
                        
                        # Update metrics
                        reps_metric.metric("Reps", st.session_state.deadlift_analysis.counter)
                        
                        if prediction:
                            form_text = prediction.replace('d_', '').replace('_', ' ').title()
                            form_metric.metric("Form", form_text)
                            confidence_metric.metric("Confidence", f"{confidence:.2f}")
                        else:
                            form_metric.metric("Form", "Not Detected")
                            confidence_metric.metric("Confidence", "0.00")
                        
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
        **Deadlift Form Analysis using AI**
        
        This application uses:
        - **MediaPipe** for pose detection and landmark extraction
        - **Machine Learning** model trained on deadlift movement patterns
        - **Real-time analysis** of form and rep counting
        - **Audio Feedback** for real-time form corrections
        
        **Form Classifications:**
        - ✅ **Correct Up/Down**: Proper deadlift form
        - ❌ **Grip Issues**: Bar grip too narrow or wide
        - ❌ **Spine Issues**: Non-neutral spine position
        
        **Audio Feedback:**
        - 🔊 Audio alerts only for form errors
        - 🔇 Silent operation for correct form
        - 📢 Grip and posture corrections only
        
        **Tips for best results:**
        - Ensure good lighting
        - Position yourself fully in frame
        - Wear contrasting colors
        - Maintain steady camera position
        - Use headphones for better audio feedback
        """)
    
    # Audio status indicator
    if audio_enabled and audio_feedback and audio_feedback.audio_enabled:
        st.sidebar.success("🔊 Audio System Active")
    elif audio_enabled:
        st.sidebar.warning("🔇 Audio System Unavailable")

if __name__ == "__main__":
    main()

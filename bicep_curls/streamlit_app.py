import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from utils import extract_important_keypoints
import warnings
warnings.filterwarnings('ignore')
from Bicep_module import BicepPoseAnalysis
import tempfile
import os
import time
import pygame
import threading

# Configure page
st.set_page_config(
    page_title="Bicep Curl Form Analyzer",
    page_icon="💪",
    layout="wide"
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for audio feedback
@st.cache_resource
def initialize_audio():
    """Initialize pygame mixer for audio feedback"""
    try:
        pygame.mixer.init()
        return True
    except pygame.error as e:
        st.warning(f"Could not initialize audio: {e}")
        return False

# Audio feedback class
class AudioFeedback:
    def __init__(self):
        self.audio_enabled = initialize_audio()
        self.audio_files = {
            "PEAK_CONTRACTION": "./audio/peak_contraction.mp3",
            "LOOSE_UPPER_ARM": "./audio/loose_upper_arm.mp3",
            "LEAN_BACK": "./audio/lean_back.mp3"
        }
        self.last_error_counts = {
            'left': {"PEAK_CONTRACTION": 0, "LOOSE_UPPER_ARM": 0},
            'right': {"PEAK_CONTRACTION": 0, "LOOSE_UPPER_ARM": 0}
        }
        self.last_posture = "C"
        
    def play_audio(self, audio_file):
        """Play audio file in a separate thread to avoid blocking"""
        if not self.audio_enabled:
            return
            
        def play():
            try:
                if os.path.exists(audio_file):
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
            except pygame.error:
                pass  # Silently handle audio errors
                
        threading.Thread(target=play, daemon=True).start()
    
    def check_and_play_error_audio(self, analysis_data):
        """Check for new errors and play corresponding audio"""
        if not self.audio_enabled:
            return
            
        # Check left arm errors
        left_errors = analysis_data['left_errors']
        for error_type in ['PEAK_CONTRACTION', 'LOOSE_UPPER_ARM']:
            if left_errors[error_type] > self.last_error_counts['left'][error_type]:
                self.play_audio(self.audio_files[error_type])
                
        # Check right arm errors
        right_errors = analysis_data['right_errors']
        for error_type in ['PEAK_CONTRACTION', 'LOOSE_UPPER_ARM']:
            if right_errors[error_type] > self.last_error_counts['right'][error_type]:
                self.play_audio(self.audio_files[error_type])
        
        # Check posture change (lean back)
        if analysis_data['posture'] != "C" and self.last_posture == "C":
            self.play_audio(self.audio_files["LEAN_BACK"])
        
        # Update last known error counts
        self.last_error_counts['left'] = left_errors.copy()
        self.last_error_counts['right'] = right_errors.copy()
        self.last_posture = analysis_data['posture']

# Constants
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Load models
@st.cache_resource
def load_models():
    try:
        with open("./model/input_scaler.pkl", "rb") as f:
            input_scaler = pickle.load(f)
        with open("./model/KNN_model.pkl", "rb") as f:
            sklearn_model = pickle.load(f)
        return input_scaler, sklearn_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None

def initialize_analysis_objects():
    """Initialize bicep pose analysis objects"""
    VISIBILITY_THRESHOLD = 0.65
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
    
    left_arm_analysis = BicepPoseAnalysis(
        side="left", 
        stage_down_threshold=STAGE_DOWN_THRESHOLD, 
        stage_up_threshold=STAGE_UP_THRESHOLD, 
        peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, 
        loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, 
        visibility_threshold=VISIBILITY_THRESHOLD
    )
    
    right_arm_analysis = BicepPoseAnalysis(
        side="right", 
        stage_down_threshold=STAGE_DOWN_THRESHOLD, 
        stage_up_threshold=STAGE_UP_THRESHOLD, 
        peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, 
        loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, 
        visibility_threshold=VISIBILITY_THRESHOLD
    )
    
    return left_arm_analysis, right_arm_analysis

def process_frame(image, pose, left_arm_analysis, right_arm_analysis, input_scaler, sklearn_model):
    """Process a single frame for pose analysis"""
    POSTURE_ERROR_THRESHOLD = 0.7
    video_dimensions = [1280, 720]
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return image, None
    
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
        
        # Analyze poses
        (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
        (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
        
        # Extract keypoints and make prediction
        row = extract_important_keypoints(results, IMPORTANT_LMS)
        X = pd.DataFrame([row], columns=HEADERS[1:])
        X = pd.DataFrame(input_scaler.transform(X))
        
        predicted_class = sklearn_model.predict(X)[0]
        prediction_probabilities = sklearn_model.predict_proba(X)[0]
        class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)
        
        posture = predicted_class if class_prediction_probability >= POSTURE_ERROR_THRESHOLD else "C"
        
        # # Add overlay information
        # cv2.rectangle(image, (0, 0), (500, 40), (245, 117, 16), -1)
        
        # # Display counters
        # cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # # Display errors
        # cv2.putText(image, "R_PC", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image, "R_LUA", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.putText(image, "L_PC", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (295, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image, "L_LUA", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (375, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # # Posture info
        # cv2.putText(image, "LB", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(f"{posture}, {predicted_class}, {class_prediction_probability}"), (440, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        # # Display angles
        # if left_arm_analysis.is_visible:
        #     cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # if right_arm_analysis.is_visible:
        #     cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        #     cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Return analysis data
        analysis_data = {
            'left_counter': left_arm_analysis.counter if left_arm_analysis.is_visible else 0,
            'right_counter': right_arm_analysis.counter if right_arm_analysis.is_visible else 0,
            'left_errors': left_arm_analysis.detected_errors.copy(),
            'right_errors': right_arm_analysis.detected_errors.copy(),
            'posture': posture,
            'prediction_confidence': class_prediction_probability
        }
        
        return image, analysis_data
        
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return image, None

def main():
    st.title("💪 Bicep Curl Form Analyzer")
    st.markdown("Upload a video or use your webcam to analyze bicep curl form and count repetitions!")
    
    # Load models
    input_scaler, sklearn_model = load_models()
    if input_scaler is None or sklearn_model is None:
        st.stop()
    
    # Initialize analysis objects
    left_arm_analysis, right_arm_analysis = initialize_analysis_objects()
    
    # Initialize audio feedback
    audio_feedback = AudioFeedback()
    
    # Sidebar controls
    st.sidebar.title("Settings")
    analysis_mode = st.sidebar.radio("Choose Analysis Mode", ["Video Upload", "Webcam (Live)"])
    
    # Audio settings
    if audio_feedback.audio_enabled:
        st.sidebar.success("🔊 Audio feedback enabled")
        st.sidebar.info("Audio will play when errors are detected in real-time mode")
    else:
        st.sidebar.warning("🔇 Audio feedback disabled")
        st.sidebar.info("Install pygame for audio feedback: pip install pygame")
    
    if analysis_mode == "Video Upload":
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Processed Video")
                video_placeholder = st.empty()
            
            with col2:
                st.subheader("Analysis Results")
                stats_placeholder = st.empty()
            
            # Process video button
            if st.button("Analyze Video"):
                cap = cv2.VideoCapture(tfile.name)
                
                with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every nth frame for performance
                        if frame_count % 3 == 0:
                            processed_frame, analysis_data = process_frame(
                                frame, pose, left_arm_analysis, right_arm_analysis, 
                                input_scaler, sklearn_model
                            )
                            
                            # Display processed frame
                            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                            
                            # Display stats
                            if analysis_data:
                                with stats_placeholder.container():
                                    st.metric("Left Arm Reps", analysis_data['left_counter'])
                                    st.metric("Right Arm Reps", analysis_data['right_counter'])
                                    st.write("**Left Arm Errors:**")
                                    st.write(f"Peak Contraction: {analysis_data['left_errors']['PEAK_CONTRACTION']}")
                                    st.write(f"Loose Upper Arm: {analysis_data['left_errors']['LOOSE_UPPER_ARM']}")
                                    st.write("**Right Arm Errors:**")
                                    st.write(f"Peak Contraction: {analysis_data['right_errors']['PEAK_CONTRACTION']}")
                                    st.write(f"Loose Upper Arm: {analysis_data['right_errors']['LOOSE_UPPER_ARM']}")
                                    st.write(f"**Posture:** {analysis_data['posture']}")
                                    st.write(f"**Confidence:** {analysis_data['prediction_confidence']}")
                        
                        frame_count += 1
                
                cap.release()
                # Clean up temp file with error handling for Windows
                try:
                    import time
                    time.sleep(0.1)  # Small delay to ensure file handle is released
                    os.unlink(tfile.name)
                except PermissionError:
                    # File is still in use, mark for deletion on reboot (Windows)
                    st.warning("Temporary file cleanup will happen automatically.")
    
    elif analysis_mode == "Webcam (Live)":
        st.subheader("Live Webcam Analysis")
        st.write("Click 'Start Analysis' to begin live analysis using your webcam.")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("Live Stats")
            stats_placeholder = st.empty()
        
        start_button = st.button("Start Analysis")
        stop_button = st.button("Stop Analysis")
        
        if start_button:
            # Reset audio feedback for new session
            audio_feedback = AudioFeedback()
            
            cap = cv2.VideoCapture(0)
            
            with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                    
                    processed_frame, analysis_data = process_frame(
                        frame, pose, left_arm_analysis, right_arm_analysis, 
                        input_scaler, sklearn_model
                    )
                    
                    # Display processed frame
                    video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    # Display stats and handle audio feedback
                    if analysis_data:
                        # Play audio feedback for new errors
                        audio_feedback.check_and_play_error_audio(analysis_data)
                        
                        with stats_placeholder.container():
                            st.metric("Left Arm Reps", analysis_data['left_counter'])
                            st.metric("Right Arm Reps", analysis_data['right_counter'])
                            
                            with st.expander("Error Details"):
                                col_left, col_right = st.columns(2)
                                with col_left:
                                    st.write("**Left Arm:**")
                                    st.write(f"Peak Contraction: {analysis_data['left_errors']['PEAK_CONTRACTION']}")
                                    st.write(f"Loose Upper Arm: {analysis_data['left_errors']['LOOSE_UPPER_ARM']}")
                                with col_right:
                                    st.write("**Right Arm:**")
                                    st.write(f"Peak Contraction: {analysis_data['right_errors']['PEAK_CONTRACTION']}")
                                    st.write(f"Loose Upper Arm: {analysis_data['right_errors']['LOOSE_UPPER_ARM']}")
                            
                            st.write(f"**Posture:** {analysis_data['posture']}")
                            st.progress(analysis_data['prediction_confidence'])
                            st.write(f"Confidence: {analysis_data['prediction_confidence']}")
                    
                    if stop_button:
                        break
            
            cap.release()
    
    # Information section
    with st.expander("ℹ️ How to Use"):
        st.write("""
        **Video Upload Mode:**
        1. Upload a video file containing bicep curl exercises
        2. Click 'Analyze Video' to process the video
        3. View the processed video with pose detection overlays
        4. Check the analysis results for rep counts and form errors
        
        **Webcam Mode (with Audio Feedback):**
        1. Click 'Start Analysis' to begin live analysis
        2. Position yourself in front of the camera
        3. Perform bicep curls and see real-time feedback
        4. Listen for audio cues when form errors are detected:
           - Peak contraction errors
           - Loose upper arm movement
           - Poor posture (lean back)
        5. Click 'Stop Analysis' to end the session
        
        **Analysis Information:**
        - **Reps:** Number of completed bicep curls for each arm
        - **Peak Contraction Error:** Indicates if you're not fully contracting at the top
        - **Loose Upper Arm Error:** Indicates if your upper arm is moving too much
        - **Posture:** Overall body posture analysis (C = Correct)
        - **Confidence:** How confident the model is in its posture prediction
        
        **Audio Feedback:**
        Audio feedback is available during live webcam analysis and will play when:
        - A new peak contraction error is detected
        - A new loose upper arm error is detected  
        - Your posture changes from correct to incorrect (lean back)
        """)

if __name__ == "__main__":
    main()

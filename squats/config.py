"""
Configuration file for Squat Detection System

This file contains all the configurable parameters for the squat detection
and form analysis system.
"""

# Model Configuration
MODEL_PATH = "./model/LR_model.pkl"

# Detection Thresholds
PREDICTION_PROBABILITY_THRESHOLD = 0.6  # Lowered from 0.7 for better rep detection
VISIBILITY_THRESHOLD = 0.6

# Form Analysis Thresholds (based on analysis from analyze_bad_pose.ipynb)
# Foot placement: ratio of foot width to shoulder width
FOOT_SHOULDER_RATIO_THRESHOLDS = {
    "min": 1.2,  # Below this is "too tight"
    "max": 2.8   # Above this is "too wide"
}

# Knee placement: ratio of knee width to foot width for different squat stages
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": {
        "min": 0.4,  # Lenient but reasonable for standing position
        "max": 1.3   # Allows wider stance when standing
    },
    "middle": {
        "min": 0.6,  # Moderate
        "max": 1.0
    },
    "down": {
        "min": 0.75, # More restrictive - stricter than original 0.7
        "max": 1.05  # More restrictive - stricter than original 1.1
    }
}

# MediaPipe Configuration
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Video Processing
DEFAULT_FRAME_RESIZE_PERCENT = 50
DEFAULT_OUTPUT_FPS = 30

# Display Colors (BGR format)
COLORS = {
    "correct": (0, 255, 0),      # Green
    "incorrect": (0, 0, 255),    # Red
    "unknown": (0, 255, 255),    # Yellow
    "background": (245, 117, 16), # Orange
    "text": (255, 255, 255),     # White
    "text_dark": (0, 0, 0)       # Black
}

# Important Landmarks for Pose Detection
IMPORTANT_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER", 
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# Feedback Messages
FEEDBACK_MESSAGES = {
    "foot_placement": {
        -1: "Foot position unclear",
        0: "Good foot placement",
        1: "Feet too close together", 
        2: "Feet too far apart"
    },
    "knee_placement": {
        -1: "Knee position unclear",
        0: "Good knee alignment",
        1: "Knees caving inward",
        2: "Knees too wide"
    }
}

#!/usr/bin/env python3
"""
Squat Detection and Form Analysis System

This script provides real-time squat detection and form analysis using MediaPipe
pose estimation and machine learning models. It can count squats and provide
feedback on foot placement and knee positioning.

Usage:
    python main.py [--video VIDEO_PATH] [--webcam] [--output OUTPUT_PATH]

Examples:
    python main.py --webcam                    # Use webcam input
    python main.py --video demo.mp4            # Use video file
    python main.py --video demo.mp4 --output analyzed_video.mp4  # Save output
"""

import argparse
import sys
import os
import math
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils import rescale_frame

# Import configuration
try:
    from config import (
        PREDICTION_PROBABILITY_THRESHOLD,
        VISIBILITY_THRESHOLD,
        FOOT_SHOULDER_RATIO_THRESHOLDS,
        KNEE_FOOT_RATIO_THRESHOLDS,
        MEDIAPIPE_DETECTION_CONFIDENCE,
        MEDIAPIPE_TRACKING_CONFIDENCE
    )
except ImportError:
    # Fallback to hardcoded values if config.py is not available
    PREDICTION_PROBABILITY_THRESHOLD = 0.7
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = {"min": 1.2, "max": 2.8}
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": {"min": 0.5, "max": 1.0},
        "middle": {"min": 0.7, "max": 1.0}, 
        "down": {"min": 0.7, "max": 1.1}
    }
    MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class SquatDetector:
    """Real-time squat detection and form analysis system."""
    
    def __init__(self, model_path: str = "./model/LR_model.pkl"):
        """
        Initialize the squat detector.
        
        Args:
            model_path: Path to the trained machine learning model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Important landmarks for pose detection
        self.IMPORTANT_LMS = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
        ]
        
        # Create headers for feature extraction
        self.headers = ["label"]
        for lm in self.IMPORTANT_LMS:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", 
                           f"{lm.lower()}_z", f"{lm.lower()}_v"]
        
        # Detection parameters
        self.PREDICTION_PROB_THRESHOLD = PREDICTION_PROBABILITY_THRESHOLD
        self.VISIBILITY_THRESHOLD = VISIBILITY_THRESHOLD
        
        # Form analysis thresholds (based on analysis from analyze_bad_pose.ipynb)
        self.FOOT_SHOULDER_RATIO_THRESHOLDS = FOOT_SHOULDER_RATIO_THRESHOLDS
        self.KNEE_FOOT_RATIO_THRESHOLDS = KNEE_FOOT_RATIO_THRESHOLDS
        
        # State variables
        self.counter = 0
        self.current_stage = ""
        self.show_debug = False
        
    def load_model(self):
        """Load the trained machine learning model."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def extract_important_keypoints(self, results) -> List[float]:
        """
        Extract important keypoints from MediaPipe pose detection.
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            Flattened list of keypoint coordinates and visibility
        """
        landmarks = results.pose_landmarks.landmark
        data = []
        
        for lm in self.IMPORTANT_LMS:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
        
        return np.array(data).flatten().tolist()
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance between the points
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def analyze_foot_knee_placement(self, results, stage: str) -> Dict[str, float]:
        """
        Analyze foot and knee placement during squat.
        
        Args:
            results: MediaPipe pose detection results
            stage: Current squat stage ("up", "middle", "down")
            
        Returns:
            Dictionary with analysis results:
            -1: Unknown (poor visibility)
            0: Correct placement
            1: Too tight
            2: Too wide
            
            Also includes ratio values for display
        """
        analysis_results = {
            "foot_placement": -1,
            "knee_placement": -1,
            "foot_shoulder_ratio": 0.0,
            "knee_foot_ratio": 0.0,
        }
        
        if not results.pose_landmarks:
            return analysis_results
            
        landmarks = results.pose_landmarks.landmark
        
        # Check visibility of key landmarks
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        
        for landmark in required_landmarks:
            if landmarks[landmark.value].visibility < self.VISIBILITY_THRESHOLD:
                return analysis_results
        
        # Calculate shoulder width
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        
        # Calculate foot width
        left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        foot_width = self.calculate_distance(left_foot, right_foot)
        
        # Analyze foot placement
        foot_shoulder_ratio = foot_width / shoulder_width
        analysis_results["foot_shoulder_ratio"] = foot_shoulder_ratio
        
        min_ratio = self.FOOT_SHOULDER_RATIO_THRESHOLDS["min"]
        max_ratio = self.FOOT_SHOULDER_RATIO_THRESHOLDS["max"]
        
        if min_ratio <= foot_shoulder_ratio <= max_ratio:
            analysis_results["foot_placement"] = 0  # Correct
        elif foot_shoulder_ratio < min_ratio:
            analysis_results["foot_placement"] = 1  # Too tight
        else:
            analysis_results["foot_placement"] = 2  # Too wide
        
        # Calculate knee width
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        knee_width = self.calculate_distance(left_knee, right_knee)
        
        # Analyze knee placement based on stage
        knee_foot_ratio = knee_width / foot_width
        analysis_results["knee_foot_ratio"] = knee_foot_ratio
        
        # Analyze knee placement for both UP and DOWN stages
        for stage_name in ["up", "down"]:
            if stage_name in self.KNEE_FOOT_RATIO_THRESHOLDS:
                thresholds = self.KNEE_FOOT_RATIO_THRESHOLDS[stage_name]
                min_knee_ratio = thresholds["min"]
                max_knee_ratio = thresholds["max"]
                
                if self.show_debug:
                    print(f"Knee analysis - Stage: {stage_name}, Ratio: {knee_foot_ratio:.3f}, Range: [{min_knee_ratio}-{max_knee_ratio}]")
                
                # Determine placement for this stage
                if min_knee_ratio <= knee_foot_ratio <= max_knee_ratio:
                    placement = 0  # Correct
                    if self.show_debug:
                        print(f"  -> {stage_name.upper()} stage knee placement: CORRECT")
                elif knee_foot_ratio < min_knee_ratio:
                    placement = 1  # Too tight
                    if self.show_debug:
                        print(f"  -> {stage_name.upper()} stage knee placement: TOO TIGHT ({knee_foot_ratio:.3f} < {min_knee_ratio})")
                else:
                    placement = 2  # Too wide
                    if self.show_debug:
                        print(f"  -> {stage_name.upper()} stage knee placement: TOO WIDE ({knee_foot_ratio:.3f} > {max_knee_ratio})")
                
                # Store results for both stages
                analysis_results[f"knee_placement_{stage_name}"] = placement
        
        # For backward compatibility, use the current/predicted stage for the main knee_placement
        analysis_stage = stage if stage else "up"  # Default to UP if no stage
        if analysis_stage in self.KNEE_FOOT_RATIO_THRESHOLDS:
            analysis_results["knee_placement"] = analysis_results.get(f"knee_placement_{analysis_stage}", -1)
        else:
            analysis_results["knee_placement"] = -1
            if self.show_debug:
                print(f"Knee analysis - Stage '{analysis_stage}' not in thresholds, available stages: {list(self.KNEE_FOOT_RATIO_THRESHOLDS.keys())}")
        
        return analysis_results
    
    def predict_squat_stage(self, results) -> Tuple[str, float]:
        """
        Predict the current squat stage using the trained model.
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            Tuple of (predicted_stage, confidence)
        """
        try:
            # Extract keypoints
            row = self.extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=self.headers[1:])
            
            # Make prediction
            predicted_class = self.model.predict(X)[0]
            predicted_class = "down" if predicted_class == 0 else "up"
            
            # Get prediction probability
            prediction_probs = self.model.predict_proba(X)[0]
            confidence = round(prediction_probs[prediction_probs.argmax()], 2)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "unknown", 0.0
    
    def update_counter(self, predicted_stage: str, confidence: float):
        """
        Update squat counter based on stage transitions.
        
        Args:
            predicted_stage: Predicted squat stage
            confidence: Prediction confidence
        """
        if self.show_debug:
            print(f"Update counter - Predicted: {predicted_stage} ({confidence}), Current: {self.current_stage}, Threshold: {self.PREDICTION_PROB_THRESHOLD}")
            
        # Only update stage if confidence is high enough
        if confidence >= self.PREDICTION_PROB_THRESHOLD:
            if predicted_stage == "down":
                if self.current_stage != "down":
                    if self.show_debug:
                        print(f"Stage transition: {self.current_stage} -> DOWN")
                self.current_stage = "down"
                
            elif predicted_stage == "up":
                # Count a rep when transitioning from DOWN to UP
                if self.current_stage == "down":
                    self.counter += 1
                    if self.show_debug:
                        print(f"REP COMPLETED! Count: {self.counter} (DOWN -> UP)")
                elif self.show_debug and self.current_stage != "up":
                    print(f"Stage transition: {self.current_stage} -> UP (no count - not from DOWN)")
                
                self.current_stage = "up"
        elif self.show_debug:
            print(f"Low confidence ({confidence} < {self.PREDICTION_PROB_THRESHOLD}) - no stage update")
    
    def get_placement_text(self, placement_code: int, stage: str = "", ratio: float = 0.0) -> str:
        """Convert placement code to human-readable text with detailed feedback."""
        if placement_code == -1:
            return "Unknown"
        elif placement_code == 0:
            return "Correct"
        elif placement_code == 1:
            if stage:
                return f"Too tight ({ratio:.2f})"
            return "Too tight"
        elif placement_code == 2:
            if stage:
                return f"Too wide ({ratio:.2f})"
            return "Too wide"
        else:
            return "Error"
    
    def draw_info(self, image: np.ndarray, predicted_stage: str, confidence: float,
                 analysis_results: Dict[str, float], analysis_stage: str = ""):
        """
        Draw information overlay on the image.
        
        Args:
            image: Input image
            predicted_stage: Predicted squat stage
            confidence: Prediction confidence
            analysis_results: Results from form analysis including ratios
            analysis_stage: The actual stage used for knee analysis
        """
        foot_placement = analysis_results.get("foot_placement", -1)
        knee_placement = analysis_results.get("knee_placement", -1)
        foot_shoulder_ratio = analysis_results.get("foot_shoulder_ratio", 0.0)
        knee_foot_ratio = analysis_results.get("knee_foot_ratio", 0.0)
        
        # Status box background
        cv2.rectangle(image, (0, 0), (700, 100), (245, 117, 16), -1)
        
        # Counter and stage
        cv2.putText(image, "COUNT", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'{self.counter}', (10, 45), cv2.FONT_HERSHEY_COMPLEX, 
                   1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage and confidence
        cv2.putText(image, "STAGE", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'{predicted_stage} ({confidence})', (100, 45), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Foot placement with ratio
        cv2.putText(image, "FOOT", (300, 20), cv2.FONT_HERSHEY_COMPLEX, 
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        foot_text = self.get_placement_text(foot_placement, "foot", foot_shoulder_ratio)
        foot_color = (0, 255, 0) if foot_placement == 0 else (0, 0, 255)
        cv2.putText(image, foot_text, (300, 45), cv2.FONT_HERSHEY_COMPLEX, 
                   0.7, foot_color, 2, cv2.LINE_AA)
        
        # Show knee analysis for both UP and DOWN stages only
        knee_up = analysis_results.get("knee_placement_up", -1)
        knee_down = analysis_results.get("knee_placement_down", -1)
        
        # Knee header
        cv2.putText(image, "KNEE", (500, 20), cv2.FONT_HERSHEY_COMPLEX, 
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # UP stage analysis
        up_color = (0, 255, 0) if knee_up == 0 else (0, 0, 255) if knee_up > 0 else (128, 128, 128)
        up_text = self.get_placement_text(knee_up)
        cv2.putText(image, f"UP: {up_text}", (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, up_color, 1, cv2.LINE_AA)
        
        # DOWN stage analysis  
        down_color = (0, 255, 0) if knee_down == 0 else (0, 0, 255) if knee_down > 0 else (128, 128, 128)
        down_text = self.get_placement_text(knee_down)
        cv2.putText(image, f"DOWN: {down_text}", (500, 65), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, down_color, 1, cv2.LINE_AA)
        
        # Instructions and debug info
        cv2.putText(image, "Press 'q' to quit, 'r' to reset counter, 's' to show debug", 
                   (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Show thresholds for current stage (debug info)
        if hasattr(self, 'show_debug') and self.show_debug:
            debug_y = image.shape[0] - 80  # More space needed for both stages
            
            # Show foot analysis
            foot_min = self.FOOT_SHOULDER_RATIO_THRESHOLDS["min"]
            foot_max = self.FOOT_SHOULDER_RATIO_THRESHOLDS["max"]
            cv2.putText(image, f"Foot: {foot_shoulder_ratio:.2f} [{foot_min}-{foot_max}]", 
                       (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Show knee analysis for both UP and DOWN stages
            for i, stage_name in enumerate(["up", "down"]):
                if stage_name in self.KNEE_FOOT_RATIO_THRESHOLDS:
                    thresholds = self.KNEE_FOOT_RATIO_THRESHOLDS[stage_name]
                    # Color code: current stage in yellow, other stage in cyan
                    color = (255, 255, 0) if stage_name == analysis_stage else (255, 255, 128)
                    y_pos = debug_y - 20 - (i * 15)
                    cv2.putText(image, f"Knee/{stage_name.upper()}: {knee_foot_ratio:.2f} [{thresholds['min']}-{thresholds['max']}]", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.45, color, 1, cv2.LINE_AA)
    
    def process_frame(self, image: np.ndarray, pose) -> np.ndarray:
        """
        Process a single frame for squat detection and analysis.
        
        Args:
            image: Input image frame
            pose: MediaPipe pose instance
            
        Returns:
            Processed image with annotations
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process pose
        results = pose.process(image_rgb)
        
        # Convert back to BGR
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Default values
        predicted_stage = "unknown"
        confidence = 0.0
        analysis_stage = "up"  # Default analysis stage
        analysis_results = {
            "foot_placement": -1,
            "knee_placement": -1,
            "foot_shoulder_ratio": 0.0,
            "knee_foot_ratio": 0.0,
        }
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
            )
            
            try:
                # Predict squat stage
                predicted_stage, confidence = self.predict_squat_stage(results)
                
                # Update counter
                self.update_counter(predicted_stage, confidence)
                
                # Use current stage for analysis, fallback to predicted stage
                analysis_stage = self.current_stage if self.current_stage else predicted_stage
                if not analysis_stage:  # Extra safety check
                    analysis_stage = "up"
                
                # Analyze form using the determined stage
                analysis_results = self.analyze_foot_knee_placement(results, analysis_stage)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                analysis_stage = "up"  # Default fallback on error
        else:
            # No pose landmarks detected - use current stage or default to "up"
            analysis_stage = self.current_stage if self.current_stage else "up"
        
        # Draw information overlay with the actual analysis stage used
        self.draw_info(image, predicted_stage, confidence, analysis_results, analysis_stage)
        
        return image
    
    def reset_counter(self):
        """Reset the squat counter."""
        self.counter = 0
        self.current_stage = ""
        print("Counter reset!")
        
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
        print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")


def main():
    """Main function to run the squat detection system."""
    parser = argparse.ArgumentParser(description="Squat Detection and Form Analysis")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--model", type=str, default="./model/LR_model.pkl", 
                       help="Path to trained model")
    parser.add_argument("--resize", type=int, default=50, 
                       help="Resize frame percentage (default: 50)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not args.webcam:
        print("Error: Please specify either --video or --webcam")
        sys.exit(1)
    
    if args.video and args.webcam:
        print("Error: Please specify either --video or --webcam, not both")
        sys.exit(1)
    
    # Initialize squat detector
    detector = SquatDetector(model_path=args.model)
    
    # Initialize video capture
    if args.webcam:
        cap = cv2.VideoCapture(0)
        print("Using webcam input...")
    else:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print(f"Using video file: {args.video}")
    
    # Check if video capture is successful
    if not cap.isOpened():
        print("Error: Could not open video source")
        sys.exit(1)
    
    # Initialize video writer if output is specified
    out = None
    if args.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.resize / 100)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.resize / 100)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output will be saved to: {args.output}")
    
    # Initialize MediaPipe pose
    with mp_pose.Pose(
        min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE, 
        min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE
    ) as pose:
        print("\nSquat Detection Started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset counter")
        print("  's' - Toggle debug info")
        print("  Space - Pause/Resume")
        print("-" * 40)
        
        paused = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = rescale_frame(frame, args.resize)
            
            if not paused:
                # Process frame
                frame = detector.process_frame(frame, pose)
            else:
                # Show pause indicator
                cv2.putText(frame, "PAUSED - Press Space to resume", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Squat Detection", frame)
            
            # Save frame if output is specified
            if out is not None and not paused:
                out.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_counter()
            elif key == ord('s'):
                detector.toggle_debug()
            elif key == ord(' '):  # Space bar
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Fix for macOS window closing issue
    for i in range(5):
        cv2.waitKey(1)
    
    print(f"\nFinal squat count: {detector.counter}")
    print("Squat detection completed!")


if __name__ == "__main__":
    main()

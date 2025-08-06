import sys
import os
import math
import pickle
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# Import configuration
try:
    from config import (
        PREDICTION_PROBABILITY_THRESHOLD,
        VISIBILITY_THRESHOLD,
        FOOT_SHOULDER_RATIO_THRESHOLDS,
        KNEE_FOOT_RATIO_THRESHOLDS
    )
except ImportError:
    # Fallback to hardcoded values if config.py is not available
    PREDICTION_PROBABILITY_THRESHOLD = 0.6
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = {"min": 1.2, "max": 2.8}
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": {"min": 0.4, "max": 1.3},
        "down": {"min": 0.75, "max": 1.05}
    }

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
        self.counter = 0.0  # Changed to float for 0.5 increments
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
    
    def update_counter(self, predicted_stage: str, confidence: float, analysis_results: Dict = None):
        """
        Update squat counter based on stage transitions and form correctness.
        
        Args:
            predicted_stage: Predicted squat stage
            confidence: Prediction confidence
            analysis_results: Form analysis results to check correctness
        """
        if self.show_debug:
            print(f"Update counter - Predicted: {predicted_stage} ({confidence}), Current: {self.current_stage}, Threshold: {self.PREDICTION_PROB_THRESHOLD}")
            
        # Only update stage if confidence is high enough
        if confidence >= self.PREDICTION_PROB_THRESHOLD:
            if predicted_stage == "down":
                if self.current_stage != "down":
                    if self.show_debug:
                        print(f"Stage transition: {self.current_stage} -> DOWN")
                    
                    # Award 0.5 points for correct DOWN stage form
                    if analysis_results:
                        foot_correct = analysis_results.get("foot_placement", -1) == 0
                        knee_down_correct = analysis_results.get("knee_placement_down", -1) == 0
                        
                        if foot_correct and knee_down_correct:
                            self.counter += 0.5
                            if self.show_debug:
                                print(f"DOWN STAGE COMPLETED! Count: {self.counter:.1f} (+0.5 for correct DOWN form)")
                                print(f"  Form check - Foot: {'✓' if foot_correct else '✗'}, Knee DOWN: {'✓' if knee_down_correct else '✗'}")
                        else:
                            if self.show_debug:
                                print(f"DOWN stage form check FAILED - No points awarded")
                                print(f"  Form check - Foot: {'✓' if foot_correct else '✗'}, Knee DOWN: {'✓' if knee_down_correct else '✗'}")
                    else:
                        # No analysis results available - award points anyway (fallback)
                        self.counter += 0.5
                        if self.show_debug:
                            print(f"DOWN STAGE COMPLETED! Count: {self.counter:.1f} (+0.5, no form analysis available)")
                
                self.current_stage = "down"
                
            elif predicted_stage == "up":
                if self.current_stage != "up":
                    if self.show_debug:
                        print(f"Stage transition: {self.current_stage} -> UP")
                    
                    # Award 0.5 points for correct UP stage form
                    if analysis_results:
                        foot_correct = analysis_results.get("foot_placement", -1) == 0
                        knee_up_correct = analysis_results.get("knee_placement_up", -1) == 0
                        
                        if foot_correct and knee_up_correct:
                            self.counter += 0.5
                            if self.show_debug:
                                print(f"UP STAGE COMPLETED! Count: {self.counter:.1f} (+0.5 for correct UP form)")
                                print(f"  Form check - Foot: {'✓' if foot_correct else '✗'}, Knee UP: {'✓' if knee_up_correct else '✗'}")
                        else:
                            if self.show_debug:
                                print(f"UP stage form check FAILED - No points awarded")
                                print(f"  Form check - Foot: {'✓' if foot_correct else '✗'}, Knee UP: {'✓' if knee_up_correct else '✗'}")
                    else:
                        # No analysis results available - award points anyway (fallback)
                        self.counter += 0.5
                        if self.show_debug:
                            print(f"UP STAGE COMPLETED! Count: {self.counter:.1f} (+0.5, no form analysis available)")
                            
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
        cv2.putText(image, f'{self.counter:.1f}', (10, 45), cv2.FONT_HERSHEY_COMPLEX, 
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
        
        # Show knee analysis based on current stage
        knee_up = analysis_results.get("knee_placement_up", -1)
        knee_down = analysis_results.get("knee_placement_down", -1)
        
        # Knee header
        cv2.putText(image, "KNEE", (500, 20), cv2.FONT_HERSHEY_COMPLEX, 
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Show knee analysis only for the current stage
        current_stage = self.current_stage if self.current_stage else predicted_stage
        
        if current_stage == "up":
            # Show UP stage analysis when in UP stage
            up_color = (0, 255, 0) if knee_up == 0 else (0, 0, 255) if knee_up > 0 else (128, 128, 128)
            up_text = self.get_placement_text(knee_up)
            cv2.putText(image, f"UP: {up_text}", (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, up_color, 2, cv2.LINE_AA)
        elif current_stage == "down":
            # Show DOWN stage analysis when in DOWN stage
            down_color = (0, 255, 0) if knee_down == 0 else (0, 0, 255) if knee_down > 0 else (128, 128, 128)
            down_text = self.get_placement_text(knee_down)
            cv2.putText(image, f"DOWN: {down_text}", (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, down_color, 2, cv2.LINE_AA)
        else:
            # Unknown stage - show generic knee analysis
            knee_generic = analysis_results.get("knee_placement", -1)
            generic_color = (0, 255, 0) if knee_generic == 0 else (0, 0, 255) if knee_generic > 0 else (128, 128, 128)
            generic_text = self.get_placement_text(knee_generic)
            cv2.putText(image, f"KNEE: {generic_text}", (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, generic_color, 2, cv2.LINE_AA)
        
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
                
                # Use current stage for analysis, fallback to predicted stage
                analysis_stage = self.current_stage if self.current_stage else predicted_stage
                if not analysis_stage:  # Extra safety check
                    analysis_stage = "up"
                
                # Analyze form using the determined stage
                analysis_results = self.analyze_foot_knee_placement(results, analysis_stage)
                
                # Update counter with form analysis results
                self.update_counter(predicted_stage, confidence, analysis_results)
                
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
        self.counter = 0.0
        self.current_stage = ""
        print("Counter reset!")
        
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
        print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")

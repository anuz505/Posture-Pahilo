import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
from utils import calculate_angle
import pandas as pd
# Deadlift Analysis Class
class DeadliftPoseAnalysis:
    def __init__(self, visibility_threshold=0.65):
        self.counter = 0
        self.stage = "up"  # starting position (standing/upright)
        self.visibility_threshold = visibility_threshold
        self.prev_prediction = None
        self.is_visible = False
        self.predicted_class = None
        self.prediction_probability = 0.0
        
    def calculate_joint_angles(self, landmarks):
        """Calculate all joint angles for deadlift analysis"""
        try:
            # Extract coordinates
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

            # Calculate angles
            neck_angle = (calculate_angle(left_shoulder, nose, left_hip) + calculate_angle(right_shoulder, nose, right_hip)) / 2
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
            right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_ankle_angle = calculate_angle(left_knee, left_ankle, left_heel)
            right_ankle_angle = calculate_angle(right_knee, right_ankle, right_heel)

            return {
                'neck_angle': neck_angle,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle,
                'left_shoulder_angle': left_shoulder_angle,
                'right_shoulder_angle': right_shoulder_angle,
                'left_hip_angle': left_hip_angle,
                'right_hip_angle': right_hip_angle,
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'left_ankle_angle': left_ankle_angle,
                'right_ankle_angle': right_ankle_angle
            }
        except Exception as e:
            print(f"Error calculating angles: {e}")
            return None
    
    def check_visibility(self, landmarks):
        """Check if key landmarks are visible"""
        key_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        visibilities = [landmarks[lm.value].visibility for lm in key_landmarks]
        self.is_visible = all(vis > self.visibility_threshold for vis in visibilities)
        return self.is_visible
    
    def analyze_pose(self, landmarks, model, scaler=None):
        """Analyze deadlift form using the ML model"""
        # Check visibility
        if not self.check_visibility(landmarks):
            return None, None
        
        # Calculate angles
        angles = self.calculate_joint_angles(landmarks)
        if angles is None:
            return None, None
        
        # Extract ALL MediaPipe landmarks (33 landmarks total, x,y,z,visibility each)
        landmark_features = []
        for i in range(33):  # MediaPipe has 33 pose landmarks
            try:
                lm = landmarks[i]
                landmark_features.extend([lm.x, lm.y, lm.z, lm.visibility])
            except:
                # If landmark not available, use zeros
                landmark_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add calculated angles (original angles)
        angle_columns = ['neck_angle', 'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle', 'left_hip_angle', 'right_hip_angle', 'left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle']
        angle_values = [angles[col] for col in angle_columns]
        
        # For the scaled angles, use the same values as original angles (matching training data structure)
        # The training data has duplicate angle columns with .1 suffix
        scaled_angles = angle_values.copy()
        
        # Combine all features: landmarks + original angles + scaled angles (duplicated)
        all_features = landmark_features + angle_values + scaled_angles
        
        # Create column names that match training data exactly
        # 132 landmark features (x1,y1,z1,v1,...,x33,y33,z33,v33)
        landmark_cols = []
        for i in range(1, 34):  # 1 to 33
            landmark_cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        
        # Original angle columns
        original_angle_cols = angle_columns.copy()
        
        # Duplicate angle columns with .1 suffix (matching the training data structure)
        duplicate_angle_cols = [col + '.1' for col in angle_columns]
        
        # Combine all column names
        all_columns = landmark_cols + original_angle_cols + duplicate_angle_cols
        
        # Create DataFrame with raw features
        X = pd.DataFrame([all_features], columns=all_columns)
        
        # The model is a pipeline that includes StandardScaler, so just use it directly
        prediction = model.predict(X)[0]
        prediction_prob = model.predict_proba(X)[0].max()
        
        self.predicted_class = prediction
        self.prediction_probability = prediction_prob
        
        # Count reps based on model predictions
        if prediction == 'd_correct_up' and self.prev_prediction == 'd_correct_down':
            self.counter += 1
        
        # Update stage based on current prediction for display purposes
        if prediction == 'd_correct_up':
            self.stage = "up"
        elif prediction == 'd_correct_down':
            self.stage = "down"
        
        # Store current prediction for next frame
        self.prev_prediction = prediction
        
        return prediction, prediction_prob

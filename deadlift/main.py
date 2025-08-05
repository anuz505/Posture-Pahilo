import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg

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
    
    def analyze_pose(self, landmarks, frame, model):
        """Analyze deadlift form using the ML model"""
        # Check visibility
        if not self.check_visibility(landmarks):
            return None, None
        
        # Calculate angles
        angles = self.calculate_joint_angles(landmarks)
        if angles is None:
            return None, None
        
        # Extract ALL MediaPipe landmarks (33 landmarks total, x,y,z,visibility each)
        # This matches your training data structure exactly
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
        
        # Store current prediction for next frame
        self.prev_prediction = prediction
        
        return prediction, prediction_prob

# Load the trained model
try:
    with open('./model/deadlift.pkl', 'rb') as f:
        deadlift_model = pickle.load(f)
    print("Model loaded successfully!")
    
    # The model is already a pipeline with StandardScaler included
    # Extract the scaler from the pipeline for reference (optional)
    try:
        scaler = deadlift_model.named_steps['standardscaler']
        print("Scaler extracted from pipeline successfully!")
    except:
        scaler = None
        print("No separate scaler needed - using pipeline model")
        
except FileNotFoundError as e:
    print(f"Error: Model files not found in ./model/ directory! {e}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

# Initialize analysis object
deadlift_analysis = DeadliftPoseAnalysis()

# Confidence threshold for predictions
PREDICTION_THRESHOLD = 0.7

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Flip image horizontally for selfie view
        image = cv2.flip(image, 1)
        
        video_dimensions = [1280, 720]

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            # Recolor image back to BGR for OpenCV and make it writable
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Display "No pose detected" message
            cv2.putText(image, "NO POSE DETECTED", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Deadlift Analysis", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Recolor image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

        # Make detection
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get model prediction
            prediction, confidence = deadlift_analysis.analyze_pose(landmarks, image, deadlift_model)
            
            if prediction and confidence > PREDICTION_THRESHOLD:
                # Display results
                
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
                
            else:
                # Low confidence prediction
                cv2.rectangle(image, (0, 0), (400, 40), (245, 117, 16), -1)
                cv2.putText(image, "LOW CONFIDENCE PREDICTION", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error: {e}")
            cv2.putText(image, f"ERROR: {str(e)[:30]}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Display instructions
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Deadlift Analysis", image)
        
        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Clean up windows
    for i in range(1, 5):
        cv2.waitKey(1)
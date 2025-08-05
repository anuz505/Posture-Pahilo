import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os, csv
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for deadlift
# For deadlifts, we need more lower-body landmarks
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# Generate all columns of the data frame
HEADERS = ["class"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
def init_csv(dataset_path: str):
    '''
    Create a blank csv file with just columns
    '''
    # Ignore if file is already exist
    if os.path.exists(dataset_path):
        return

    # Write all the columns to a empty file
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(HEADERS)

def export_landmark_to_csv(dataset_path: str, results, action: str) -> None:
    '''
    Export Labeled Data from detected landmark to csv
    '''
    try:
        keypoints = [action]
        
        # Extract all landmark coordinates
        for lm in IMPORTANT_LMS:
            landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark[lm].value]
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        # Append new row to .csv file
        with open(dataset_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
        
        return True
    except Exception as e:
        print(f"Error exporting landmarks: {e}")
        return False

DATASET_PATH = "deadlift_landmarks.csv"

# Check if video file exists
if not os.path.exists("demo.mp4"):
    print("Error: demo.mp4 not found!")
    exit(1)

cap = cv2.VideoCapture("demo.mp4")
save_counts = 0

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video!")
    exit(1)

init_csv(DATASET_PATH)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Resize frame (corrected dimensions - landscape format)
        width = 1280
        height = 720
        image = cv2.resize(image, (width, height))

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        # Recolor back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            print("Cannot detect pose - No human found")
            # Still display the image with instructions even if no pose detected
            cv2.putText(image, f"Saved: {save_counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "u: correct up | d: correct down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "s: spine not neutral | a: arms spread", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "n: arms narrow | q: quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "NO POSE DETECTED", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Deadlift Labeling", image)
            
            # Check for quit key even when no pose detected
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            continue

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display instructions and saved count
        cv2.putText(image, f"Saved: {save_counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "u: correct up | d: correct down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "s: spine not neutral | a: arms spread", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "n: arms narrow | q: quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Deadlift Labeling", image)

        # Process keypress
        k = cv2.waitKey(1) & 0xFF
        
        # Different posture labels based on your notebook
        if k == ord('u'):  # Correct up position
            if export_landmark_to_csv(DATASET_PATH, results, "d_correct_up"):
                save_counts += 1
                print("Saved: Correct position (UP)")
                
        elif k == ord('d'):  # Correct down position
            if export_landmark_to_csv(DATASET_PATH, results, "d_correct_down"):
                save_counts += 1
                print("Saved: Correct position (DOWN)")
                
        elif k == ord('s'):  # Not spine neutral
            if export_landmark_to_csv(DATASET_PATH, results, "d_spine_neutral_down"):
                save_counts += 1
                print("Saved: Not spine neutral")
                
        elif k == ord('a'):  # Arms spread
            if export_landmark_to_csv(DATASET_PATH, results, "d_arms_spread_down"):
                save_counts += 1
                print("Saved: Arms spread")
                
        elif k == ord('n'):  # Arms narrow
            if export_landmark_to_csv(DATASET_PATH, results, "d_arms_narrow_down"):
                save_counts += 1
                print("Saved: Arms narrow")
                
        elif k == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()
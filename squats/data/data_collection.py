import mediapipe as mp
import cv2
import os
import sys

# Add parent directory to Python path to find utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils package
from utils import (
    IMPORTANT_LMS, 
    landmarks, 
    rescale_frame, 
    init_csv, 
    export_landmark_to_csv,
    describe_dataset
)

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def collect_squat_data(video_source, dataset_path: str, dataset_type: str = "train"):
    """
    Collect squat pose data from video file or camera for training/testing
    
    Args:
        video_source: Path to video file (str) or camera index (int, typically 0)
        dataset_path (str): Path where CSV dataset will be saved
        dataset_type (str): Type of dataset ("train" or "test")
    """
    
    # Handle camera vs video file
    if isinstance(video_source, int):
        print(f"Opening camera {video_source}...")
        cap = cv2.VideoCapture(video_source)
        source_name = f"Camera {video_source}"
    else:
        print(f"Opening video file: {video_source}")
        cap = cv2.VideoCapture(video_source)
        source_name = os.path.basename(video_source)
    
    # Check if camera/video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open {source_name}")
        return
    
    up_save_count = 0
    down_save_count = 0

    # Initialize CSV file with headers
    init_csv(dataset_path)
    
    print(f"Starting {dataset_type} data collection from {source_name}...")
    print("Controls:")
    print("Press 'u' - Save current pose as UP position")
    print("Press 'd' - Save current pose as DOWN position") 
    print("Press 'q' - Quit and save dataset")
    print("-" * 50)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                print("End of video reached or cannot read frame")
                break
            
            # Reduce size of frame for better performance
            image = rescale_frame(image, 60)
            
            # Flip image horizontally for mirror effect (optional for training data)
            if dataset_type == "train":
                image = cv2.flip(image, 1)

            # Convert BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process pose detection
            results = pose.process(image)

            # Convert RGB back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks and connections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Display counters and instructions
            cv2.putText(image, f"UP saved: {up_save_count}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"DOWN saved: {down_save_count}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Press 'u'=UP, 'd'=DOWN, 'q'=QUIT", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the frame
            cv2.imshow(f"Squat Data Collection - {dataset_type.upper()}", image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d') and results.pose_landmarks:
                export_landmark_to_csv(dataset_path, results, "down")
                down_save_count += 1
                print(f"DOWN position saved! Total: {down_save_count}")
                
            elif key == ord("u") and results.pose_landmarks:
                export_landmark_to_csv(dataset_path, results, "up")
                up_save_count += 1
                print(f"UP position saved! Total: {up_save_count}")
                
            elif key == ord("q"):
                print("Quitting data collection...")
                break

    cap.release()
    cv2.destroyAllWindows()

    # Fix potential window closing issues on some systems
    for i in range(1, 5):
        cv2.waitKey(1)
    
    print(f"\nData collection completed!")
    print(f"Total UP positions saved: {up_save_count}")
    print(f"Total DOWN positions saved: {down_save_count}")
    print(f"Dataset saved to: {dataset_path}")
    
    # Display dataset summary
    if os.path.exists(dataset_path):
        print("\nDataset Summary:")
        describe_dataset(dataset_path)


def main():
    """Main function to run data collection"""
    
    # Configuration
    train_video_path = "demo.mp4"  # Update path as needed
    test_video_path = "../data/squat/squat_test_4.mp4"    # Update path as needed
    
    train_dataset_path = "./train.csv"
    test_dataset_path = "./test.csv"
    
    print("Squat Pose Data Collection Tool")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Collect training data from video file")
        print("2. Collect training data from camera")
        print("3. Collect test data from video file") 
        print("4. Collect test data from camera")
        print("5. View existing dataset summary")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            if os.path.exists(train_video_path):
                collect_squat_data(train_video_path, train_dataset_path, "train")
            else:
                print(f"Training video not found at: {train_video_path}")
                custom_path = input("Enter custom video path (or press Enter to skip): ").strip()
                if custom_path and os.path.exists(custom_path):
                    collect_squat_data(custom_path, train_dataset_path, "train")
                    
        elif choice == "2":
            camera_index = input("Enter camera index (0 for default camera, 1 for external, etc.): ").strip()
            try:
                camera_index = int(camera_index) if camera_index else 0
                collect_squat_data(camera_index, train_dataset_path, "train")
            except ValueError:
                print("Invalid camera index. Please enter a number.")
                    
        elif choice == "3":
            if os.path.exists(test_video_path):
                collect_squat_data(test_video_path, test_dataset_path, "test")
            else:
                print(f"Test video not found at: {test_video_path}")
                custom_path = input("Enter custom video path (or press Enter to skip): ").strip()
                if custom_path and os.path.exists(custom_path):
                    collect_squat_data(custom_path, test_dataset_path, "test")
                    
        elif choice == "4":
            camera_index = input("Enter camera index (0 for default camera, 1 for external, etc.): ").strip()
            try:
                camera_index = int(camera_index) if camera_index else 0
                collect_squat_data(camera_index, test_dataset_path, "test")
            except ValueError:
                print("Invalid camera index. Please enter a number.")
                    
        elif choice == "5":
            dataset_path = input("Enter dataset path (train.csv/test.csv): ").strip()
            if os.path.exists(dataset_path):
                describe_dataset(dataset_path)
            else:
                print(f"Dataset not found at: {dataset_path}")
                
        elif choice == "6":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()

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
import cv2
import mediapipe as mp
from squat_module import SquatDetector

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
    
    print(f"\nFinal squat count: {detector.counter:.1f}")
    print("Squat detection completed!")


if __name__ == "__main__":
    main()

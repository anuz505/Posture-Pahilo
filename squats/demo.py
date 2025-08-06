#!/usr/bin/env python3
"""
Demo script for Squat Detection System

This script demonstrates different ways to use the squat detection system.
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print a welcome banner."""
    print("="*60)
    print("    SQUAT DETECTION AND FORM ANALYSIS SYSTEM")
    print("="*60)
    print()

def check_requirements():
    """Check if required files exist."""
    required_files = [
        "main.py",
        "config.py", 
        "model/LR_model.pkl",
        "utils/utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def demo_webcam():
    """Demo using webcam input."""
    print("\n🎥 Starting webcam demo...")
    print("Make sure your webcam is connected and working.")
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    try:
        subprocess.run([sys.executable, "main.py", "--webcam"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running webcam demo: {e}")
    except KeyboardInterrupt:
        print("\nDemo cancelled by user.")

def demo_video():
    """Demo using video file."""
    video_files = ["squat_demo.mp4", "demo.mp4"]
    available_videos = [v for v in video_files if os.path.exists(v)]
    
    if not available_videos:
        print("\n❌ No demo video files found.")
        print("Expected files: squat_demo.mp4 or demo.mp4")
        return
    
    video_file = available_videos[0]
    print(f"\n🎬 Starting video demo with: {video_file}")
    
    try:
        subprocess.run([sys.executable, "main.py", "--video", video_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running video demo: {e}")
    except KeyboardInterrupt:
        print("\nDemo cancelled by user.")

def demo_video_with_output():
    """Demo with video output saving."""
    video_files = ["squat_demo.mp4", "demo.mp4"]
    available_videos = [v for v in video_files if os.path.exists(v)]
    
    if not available_videos:
        print("\n❌ No demo video files found.")
        return
    
    video_file = available_videos[0]
    output_file = f"analyzed_{int(time.time())}.mp4"
    
    print(f"\n💾 Demo with output saving:")
    print(f"Input: {video_file}")
    print(f"Output: {output_file}")
    
    try:
        subprocess.run([
            sys.executable, "main.py", 
            "--video", video_file,
            "--output", output_file
        ], check=True)
        
        if os.path.exists(output_file):
            print(f"✅ Output saved successfully: {output_file}")
        else:
            print("❌ Output file was not created.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
    except KeyboardInterrupt:
        print("\nDemo cancelled by user.")

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 USAGE EXAMPLES:")
    print("-" * 40)
    print("1. Use webcam:")
    print("   python main.py --webcam")
    print()
    print("2. Analyze video file:")
    print("   python main.py --video your_video.mp4")
    print()
    print("3. Save analyzed video:")
    print("   python main.py --video input.mp4 --output analyzed.mp4")
    print()
    print("4. Use different model:")
    print("   python main.py --webcam --model ./model/RF_model.pkl")
    print()
    print("5. Resize video (for better performance):")
    print("   python main.py --video input.mp4 --resize 30")
    print()

def main():
    """Main demo function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\nPlease ensure all required files are present before running demos.")
        return
    
    while True:
        print("\n🎯 DEMO OPTIONS:")
        print("1. Webcam Demo")
        print("2. Video File Demo") 
        print("3. Video Demo with Output Saving")
        print("4. Show Usage Examples")
        print("5. Exit")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                demo_webcam()
            elif choice == "2":
                demo_video()
            elif choice == "3":
                demo_video_with_output()
            elif choice == "4":
                show_usage_examples()
            elif choice == "5":
                print("\n👋 Thanks for trying the Squat Detection System!")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo session ended.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Launch script for AI Exercise Form Analyzer

This script provides a simple way to launch the main application
with proper error handling and dependency checking.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'mediapipe', 'opencv-python', 
        'numpy', 'pandas', 'scikit-learn', 'pygame'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🏋️‍♂️ AI Exercise Form Analyzer Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main_app.py"):
        print("❌ Error: main_app.py not found!")
        print("   Make sure you're running this from the repsAI directory")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies found!")
    print("🚀 Launching application...")
    print("\n" + "=" * 50)
    
    try:
        # Launch the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main_app.py",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

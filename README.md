# 🏋️‍♂️ AI Exercise Form Analyzer

A comprehensive AI-powered exercise form analysis application that uses computer vision and machine learning to analyze workout form in real-time. This application supports multiple exercises including squats, deadlifts, and bicep curls.

## 🌟 Features

### Multi-Exercise Support

- **🦵 Squats**: Real-time squat detection with form analysis
- **🏋️ Deadlifts**: Deadlift technique monitoring and safety analysis
- **💪 Bicep Curls**: Bilateral arm analysis with posture monitoring

### Advanced Analysis Capabilities

- **Real-time pose detection** using MediaPipe
- **Machine learning models** trained on exercise-specific movement patterns
- **Audio feedback** for form corrections
- **Video upload** and **live webcam** analysis modes
- **Rep counting** with granular scoring systems
- **Form error detection** and classification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for live analysis)
- Windows/macOS/Linux

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd repsAI
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main application**

   ```bash
   streamlit run main_app.py
   ```


## 📋 Requirements

```
streamlit>=1.28.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
pygame>=2.5.0
```

## 🏗️ Project Structure

```
repsAI/
├── main_app.py                 # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── bicep_curls/              # Bicep curl analysis module
│   ├── streamlit_app.py      # Bicep curl Streamlit app
│   ├── Bicep_module.py       # Core bicep analysis logic
│   ├── model/                # Trained ML models
│   ├── audio/                # Audio feedback files
│   └── utils/                # Utility functions
├── deadlift/                 # Deadlift analysis module
│   ├── streamlit_app.py      # Deadlift Streamlit app
│   ├── deadlift_module.py    # Core deadlift analysis logic
│   ├── model/                # Trained ML models
│   ├── audio/                # Audio feedback files
│   └── utils/                # Utility functions
├── squats/                   # Squat analysis module
│   ├── streamlit_app.py      # Squat Streamlit app
│   ├── squat_module.py       # Core squat analysis logic
│   ├── model/                # Trained ML models
│   └── utils/                # Utility functions
└── utils/                    # Shared utilities
    └── utils.py              # Common utility functions
```

## 🎯 How to Use

### 1. Exercise Selection

- Launch the main application
- Choose your desired exercise from the sidebar
- Review the exercise-specific features and tips

### 2. Analysis Modes

#### Video Upload Mode

- Upload a video file (MP4, AVI, MOV, MKV)
- Click "Analyze Video" to process
- View real-time analysis results and statistics

#### Live Webcam Mode

- Click "Start Analysis" to begin live analysis
- Position yourself in front of the camera
- Receive real-time form feedback and audio cues
- Click "Stop Analysis" to end the session

### 3. Form Feedback

Each exercise provides specific form analysis:

#### Squats

- ✅ **Foot placement**: Stance width analysis
- ✅ **Knee tracking**: Knee cave detection
- ✅ **Depth analysis**: Squat depth measurement
- 🎯 **0.5 point scoring**: Granular rep counting

#### Deadlifts

- ✅ **Grip analysis**: Bar grip width monitoring
- ✅ **Spine position**: Neutral spine detection
- ✅ **Movement phases**: Lift phase identification
- 🔊 **Audio alerts**: Real-time form corrections

#### Bicep Curls

- ✅ **Bilateral analysis**: Both arms monitored
- ✅ **Peak contraction**: Full contraction detection
- ✅ **Upper arm stability**: Movement minimization
- ✅ **Posture analysis**: Body position monitoring

## 🔧 Technical Details

### AI/ML Stack

- **MediaPipe**: Real-time pose estimation
- **Scikit-learn**: Exercise classification models
- **OpenCV**: Video processing and computer vision
- **NumPy/Pandas**: Data processing and analysis


### Audio Feedback

- **Pygame**: Audio playback system
- **MP3 files**: Exercise-specific audio cues
- **Real-time triggers**: Form error detection

## 💡 Tips for Best Results

### Camera Setup

- 📹 **Good lighting**: Ensure adequate illumination
- 🎨 **Contrasting colors**: Wear clothes that contrast with background
- 📐 **Camera position**: Place camera at torso level
- 🖼️ **Full body frame**: Keep entire body visible
- 🎧 **Audio**: Use headphones for better audio feedback

### Exercise Performance

- 🎯 **Face camera**: Perform exercises facing the camera
- ⚡ **Controlled movements**: Maintain steady, controlled motions
- 🔄 **Full range**: Complete full range of motion
- ⏱️ **Consistent tempo**: Maintain consistent exercise tempo

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**

   - Check camera permissions in browser
   - Ensure no other applications are using the camera
   - Try refreshing the page

2. **Model loading errors**

   - Verify all model files are present in respective directories
   - Check file permissions
   - Ensure Python packages are properly installed

3. **Audio feedback not working**

   - Install pygame: `pip install pygame`
   - Check system audio settings
   - Verify audio files are present in audio/ directories

4. **Poor pose detection**
   - Improve lighting conditions
   - Ensure full body is visible in frame
   - Wear contrasting clothing
   - Minimize background clutter



---

**Built with ❤️ using MediaPipe, Streamlit, and Machine Learning**

_Perfect your form, prevent injuries, maximize gains_ 🏋️‍♂️

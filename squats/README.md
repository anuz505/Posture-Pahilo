# Squat Detection and Form Analysis System

A real-time squat detection and form analysis system using MediaPipe pose estimation and machine learning. This system can count squats automatically and provide feedback on foot placement and knee positioning.

## Features

- ✅ **Real-time squat counting** using trained ML models
- ✅ **Form analysis** with feedback on foot and knee placement
- ✅ **Webcam and video file support**
- ✅ **Video output saving** for analysis review
- ✅ **Interactive controls** (pause, reset, quit)
- ✅ **Configurable thresholds** for form analysis

## Quick Start

### Prerequisites

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn
```

### Basic Usage

1. **Webcam Detection:**

   ```bash
   python main.py --webcam
   ```

2. **Video File Analysis:**

   ```bash
   python main.py --video your_video.mp4
   ```

3. **Save Analysis Results:**

   ```bash
   python main.py --video input.mp4 --output analyzed_output.mp4
   ```

4. **Run Interactive Demo:**
   ```bash
   python demo.py
   ```

## System Components

### Files Structure

```
squats/
├── main.py              # Main squat detection script
├── demo.py              # Interactive demo script
├── config.py            # Configuration parameters
├── model/               # Trained ML models
│   ├── LR_model.pkl     # Logistic Regression model (default)
│   ├── RF_model.pkl     # Random Forest model
│   └── ...              # Other trained models
├── utils/
│   └── utils.py         # Utility functions
└── data/
    └── ...              # Demo videos and datasets
```

### Key Components

1. **SquatDetector Class** (`main.py`)

   - Handles pose detection and analysis
   - Manages squat counting logic
   - Provides form feedback

2. **Configuration** (`config.py`)

   - Adjustable thresholds for form analysis
   - Model and detection parameters
   - Display colors and messages

3. **Demo System** (`demo.py`)
   - Interactive demo with multiple options
   - Easy testing of different scenarios

## How It Works

### 1. Pose Detection

- Uses MediaPipe to detect 9 key body landmarks
- Extracts coordinates and visibility scores
- Processes landmarks for ML model input

### 2. Squat Counting

- Trained ML model predicts "up" or "down" positions
- Counts complete squat cycles (down → up transitions)
- Uses confidence thresholds to filter predictions

### 3. Form Analysis

#### Foot Placement

- Calculates ratio of foot width to shoulder width
- **Correct range:** 1.2 - 2.8
- **Too tight:** < 1.2 (feet too close)
- **Too wide:** > 2.8 (feet too far apart)

#### Knee Positioning

- Analyzes knee-to-foot width ratio by squat stage
- **Up position:** 0.5 - 1.0 ratio
- **Middle position:** 0.7 - 1.0 ratio
- **Down position:** 0.7 - 1.1 ratio

### 4. Real-time Feedback

- Visual indicators for correct/incorrect form
- Color-coded feedback (green = good, red = needs improvement)
- Live counter and stage display

## Controls

During detection, use these keyboard controls:

- **`q`** - Quit the application
- **`r`** - Reset the squat counter
- **`Space`** - Pause/Resume detection

## Command Line Options

```bash
python main.py [options]

Options:
  --video PATH      Path to video file
  --webcam          Use webcam input
  --output PATH     Save analyzed video to file
  --model PATH      Path to ML model (default: ./model/LR_model.pkl)
  --resize PERCENT  Resize frame percentage (default: 50)
```

## Configuration

Edit `config.py` to customize:

- Detection confidence thresholds
- Form analysis parameters
- Display colors and messages
- Model paths

### Example Customization

```python
# Adjust form analysis sensitivity
FOOT_SHOULDER_RATIO_THRESHOLDS = {
    "min": 1.0,  # More strict (was 1.2)
    "max": 3.0   # More lenient (was 2.8)
}

# Change prediction confidence requirement
PREDICTION_PROBABILITY_THRESHOLD = 0.8  # Higher confidence required
```

## Model Information

The system includes several pre-trained models:

- **LR_model.pkl** - Logistic Regression (default, fast)
- **RF_model.pkl** - Random Forest (more accurate)
- **SVC_model.pkl** - Support Vector Classifier
- **KNN_model.pkl** - K-Nearest Neighbors

To use a different model:

```bash
python main.py --webcam --model ./model/RF_model.pkl
```

## Form Analysis Details

The system analyzes squat form based on research from `analyze_bad_pose.ipynb`:

### Foot Placement Analysis

- Measures distance between foot positions
- Compares to shoulder width for normalization
- Provides feedback for optimal stance width

### Knee Tracking Analysis

- Monitors knee alignment throughout squat movement
- Stage-specific analysis (up, middle, down positions)
- Detects knee valgus (knees caving in) and excessive knee abduction

## Performance Tips

1. **Ensure good lighting** for pose detection
2. **Position camera** to capture full body
3. **Use --resize option** for better performance on slower systems
4. **Wear contrasting colors** for better landmark detection
5. **Maintain clear background** when possible

## Troubleshooting

### Common Issues

1. **Model file not found**

   - Ensure `model/LR_model.pkl` exists
   - Check file path in command line arguments

2. **Poor pose detection**

   - Improve lighting conditions
   - Ensure full body is visible in frame
   - Check camera positioning

3. **Inaccurate counting**

   - Adjust `PREDICTION_PROBABILITY_THRESHOLD` in config
   - Try different models for better accuracy
   - Ensure complete squat movements

4. **Video file issues**
   - Check video file format (MP4, AVI, MOV supported)
   - Verify file path is correct
   - Ensure video contains clear human movements

### System Requirements

- Python 3.7+
- OpenCV 4.x
- MediaPipe 0.8+
- Modern CPU (webcam real-time processing)
- Camera/webcam for live detection

## License

This project is for educational and research purposes. Please ensure proper attribution when using or modifying the code.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional form analysis metrics
- Support for multiple person detection
- Mobile/edge device optimization
- Additional exercise types

---

**Note:** This system is designed for fitness guidance and education. Always consult with fitness professionals for proper exercise form and safety.

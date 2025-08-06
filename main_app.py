#!/usr/bin/env python3
"""
Main Streamlit Application for AI-Powered Exercise Form Analysis

This is the main entry point for the AI Exercise Form Analyzer application.
Users can select from different exercise types and get real-time form analysis.

Usage:
    streamlit run main_app.py

Features:
    - Exercise selection (Squats, Deadlifts, Bicep Curls)
    - Real-time form analysis using MediaPipe and ML models
    - Audio feedback for form corrections
    - Video upload and live webcam analysis
"""

import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path

# Get the main app directory at module level to avoid path issues
MAIN_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure page
st.set_page_config(
    page_title="AI Exercise Form Analyzer",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .exercise-card {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .exercise-card:hover {
        transform: translateY(-2px);
        border-color: #667eea;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .feature-list {
        list-style-type: none;
        padding: 0;
    }
    
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .feature-list li:before {
        content: "✓ ";
        color: #28a745;
        font-weight: bold;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏋️‍♂️ AI Exercise Form Analyzer</h1>
        <p>Advanced pose estimation and machine learning for perfect workout form</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for exercise selection
    st.sidebar.header("🎯 Exercise Selection")
    
    # Exercise options
    exercise_options = {
        "🦵 Squats": {
            "description": "Analyze squat form and count repetitions",
            "features": [
                "Real-time squat detection",
                "Form analysis (foot placement, knee tracking)",
                "0.5 point scoring system",
                "Audio feedback for corrections"
            ],
            "module": "squats"
        },
        "🏋️ Deadlifts": {
            "description": "Monitor deadlift technique and safety",
            "features": [
                "Grip width analysis",
                "Spine position monitoring", 
                "Movement phase detection",
                "Audio alerts for form errors"
            ],
            "module": "deadlift"
        },
        "💪 Bicep Curls": {
            "description": "Perfect your bicep curl technique",
            "features": [
                "Bilateral arm analysis",
                "Peak contraction detection",
                "Upper arm stability monitoring",
                "Posture analysis"
            ],
            "module": "bicep_curls"
        }
    }
    
    selected_exercise = st.sidebar.selectbox(
        "Choose your exercise:",
        list(exercise_options.keys()),
        index=0
    )
    
    # Display selected exercise information
    exercise_info = exercise_options[selected_exercise]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="exercise-card">
            <h2>{selected_exercise}</h2>
            <p style="font-size: 1.1rem; color: #666;">{exercise_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.subheader("🌟 Key Features")
        feature_html = "<ul class='feature-list'>"
        for feature in exercise_info['features']:
            feature_html += f"<li>{feature}</li>"
        feature_html += "</ul>"
        st.markdown(feature_html, unsafe_allow_html=True)
        
        # Launch button
        st.markdown("---")
        launch_col1, launch_col2, launch_col3 = st.columns([1, 2, 1])
        
        with launch_col2:
            if st.button(f"🚀 Launch {selected_exercise} Analyzer", type="primary", use_container_width=True):
                # Store the selected exercise in session state
                st.session_state.selected_exercise = exercise_info['module']
                st.session_state.launch_exercise = True
                st.rerun()
    
    with col2:
        # Quick stats or tips
        st.markdown("""
        <div class="stats-container">
            <h3>💡 Quick Tips</h3>
            <ul>
                <li>Ensure good lighting</li>
                <li>Wear contrasting colors</li>
                <li>Position camera at torso level</li>
                <li>Keep full body in frame</li>
                <li>Use headphones for audio feedback</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical info
        st.markdown("""
        <div class="stats-container">
            <h3>🔧 Technology Stack</h3>
            <ul>
                <li><strong>MediaPipe:</strong> Pose detection</li>
                <li><strong>Scikit-learn:</strong> ML models</li>
                <li><strong>OpenCV:</strong> Video processing</li>
                <li><strong>Streamlit:</strong> Web interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle exercise launch
    if st.session_state.get('launch_exercise', False):
        launch_exercise_app(st.session_state.selected_exercise)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🏋️‍♂️ AI Exercise Form Analyzer - Built with ❤️ using MediaPipe and Machine Learning</p>
        <p><em>Perfect your form, prevent injuries, maximize gains</em></p>
    </div>
    """, unsafe_allow_html=True)

def launch_exercise_app(exercise_module):
    """Launch the specific exercise application"""
    st.markdown("---")
    st.success(f"🚀 {exercise_module.replace('_', ' ').title()} Analyzer Loaded!")
    
    # Add back button at the top
    if st.button("⬅️ Back to Exercise Selection", key="back_button_top"):
        st.session_state.launch_exercise = False
        st.session_state.selected_exercise = None
        st.rerun()
    
    st.markdown("---")
    
    # Store original working directory and sys.path
    original_cwd = os.getcwd()
    original_sys_path = sys.path.copy()
    
    # Clean up any previous exercise modules from sys.modules to avoid conflicts
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        # Clear specific exercise-related modules that might conflict
        if (module_name in ['audio', 'utils', 'squat_module', 'deadlift_module', 'Bicep_module', 'config'] or
            module_name.startswith('audio.') or module_name.startswith('utils.') or
            module_name.startswith('squat_module.') or module_name.startswith('deadlift_module.') or
            module_name.startswith('Bicep_module.')):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    try:
        # Import and run the specific exercise app
        if exercise_module == "squats":
            # Setup squats module with proper paths
            exercise_path = os.path.join(MAIN_APP_DIR, "squats")
            utils_path = os.path.join(exercise_path, "utils")
            
            # Change to exercise directory and update paths
            os.chdir(exercise_path)
            sys.path.insert(0, exercise_path)
            sys.path.insert(0, utils_path)
            
            # Import and run squats app
            import importlib.util
            spec = importlib.util.spec_from_file_location("squats_app", os.path.join(exercise_path, "streamlit_app.py"))
            squats_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(squats_app)
            squats_app.main()
            
        elif exercise_module == "deadlift":
            # Setup deadlift module with proper paths
            exercise_path = os.path.join(MAIN_APP_DIR, "deadlift")
            utils_path = os.path.join(exercise_path, "utils")
            audio_path = os.path.join(exercise_path, "audio")
            
            # Change to exercise directory and update paths
            os.chdir(exercise_path)
            sys.path.insert(0, exercise_path)
            sys.path.insert(0, utils_path)
            sys.path.insert(0, audio_path)
            
            # Import and run deadlift app
            import importlib.util
            spec = importlib.util.spec_from_file_location("deadlift_app", os.path.join(exercise_path, "streamlit_app.py"))
            deadlift_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(deadlift_app)
            deadlift_app.main()
            
        elif exercise_module == "bicep_curls":
            # Setup bicep_curls module with proper paths
            exercise_path = os.path.join(MAIN_APP_DIR, "bicep_curls")
            utils_path = os.path.join(exercise_path, "utils")
            audio_path = os.path.join(exercise_path, "audio")
            
            # Change to exercise directory and update paths
            os.chdir(exercise_path)
            sys.path.insert(0, exercise_path)
            sys.path.insert(0, utils_path)
            sys.path.insert(0, audio_path)
            
            # Import and run bicep curls app
            import importlib.util
            spec = importlib.util.spec_from_file_location("bicep_app", os.path.join(exercise_path, "streamlit_app.py"))
            bicep_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bicep_app)
            bicep_app.main()
    
    except ImportError as e:
        st.error(f"❌ Error importing {exercise_module} module: {e}")
        st.info("Make sure all required files are in the correct directories.")
        expected_path = os.path.join(MAIN_APP_DIR, exercise_module)
        st.code(f"Expected path: {expected_path}")
        
    except Exception as e:
        st.error(f"❌ Error launching {exercise_module} analyzer: {e}")
        st.info("Please check the application logs for more details.")
        st.exception(e)
    
    finally:
        # Always restore original working directory and sys.path
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
    
    # Add back button at the bottom too
    st.markdown("---")
    if st.button("⬅️ Back to Exercise Selection", key="back_button_bottom"):
        st.session_state.launch_exercise = False
        st.session_state.selected_exercise = None
        st.rerun()

# Initialize session state
if 'launch_exercise' not in st.session_state:
    st.session_state.launch_exercise = False
if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = None

if __name__ == "__main__":
    main()

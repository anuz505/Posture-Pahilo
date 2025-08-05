import os
import pygame
import threading
import streamlit as st
# Audio feedback class
def initialize_audio():
    """Initialize pygame mixer for audio feedback"""
    try:
        pygame.mixer.init()
        return True
    except pygame.error as e:
        st.warning(f"Could not initialize audio: {e}")
        return False
class AudioFeedback:
    
    def __init__(self):
        self.audio_enabled = initialize_audio()
        self.audio_files = {
            "PEAK_CONTRACTION": "./audio/peak_contraction.mp3",
            "LOOSE_UPPER_ARM": "./audio/loose_upper_arm.mp3",
            "LEAN_BACK": "./audio/lean_back.mp3"
        }
        self.last_error_counts = {
            'left': {"PEAK_CONTRACTION": 0, "LOOSE_UPPER_ARM": 0},
            'right': {"PEAK_CONTRACTION": 0, "LOOSE_UPPER_ARM": 0}
        }
        self.last_posture = "C"
        
    def play_audio(self, audio_file):
        """Play audio file in a separate thread to avoid blocking"""
        if not self.audio_enabled:
            return
            
        def play():
            try:
                if os.path.exists(audio_file):
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
            except pygame.error:
                pass  # Silently handle audio errors
                
        threading.Thread(target=play, daemon=True).start()
    
    def check_and_play_error_audio(self, analysis_data):
        """Check for new errors and play corresponding audio"""
        if not self.audio_enabled:
            return
            
        # Check left arm errors
        left_errors = analysis_data['left_errors']
        for error_type in ['PEAK_CONTRACTION', 'LOOSE_UPPER_ARM']:
            if left_errors[error_type] > self.last_error_counts['left'][error_type]:
                self.play_audio(self.audio_files[error_type])
                
        # Check right arm errors
        right_errors = analysis_data['right_errors']
        for error_type in ['PEAK_CONTRACTION', 'LOOSE_UPPER_ARM']:
            if right_errors[error_type] > self.last_error_counts['right'][error_type]:
                self.play_audio(self.audio_files[error_type])
        
        # Check posture change (lean back)
        if analysis_data['posture'] != "C" and self.last_posture == "C":
            self.play_audio(self.audio_files["LEAN_BACK"])
        
        # Update last known error counts
        self.last_error_counts['left'] = left_errors.copy()
        self.last_error_counts['right'] = right_errors.copy()
        self.last_posture = analysis_data['posture']

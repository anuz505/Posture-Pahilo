import os
import pygame
import threading
import streamlit as st

def initialize_audio():
    """Initialize pygame mixer for audio feedback"""
    try:
        pygame.mixer.init()
        return True
    except pygame.error as e:
        st.warning(f"Could not initialize audio: {e}")
        return False

class DeadliftAudioFeedback:
    
    def __init__(self):
        self.audio_enabled = initialize_audio()
        self.audio_files = {
            "BAR_GRIP_NARROW": "./audio/bar_grip_narrow.mp3",
            "BAR_GRIP_WIDE": "./audio/bar_grip_wide.mp3", 
            "CORRECT_POSTURE": "./audio/correct_posture.mp3",
            "NON_NEUTRAL_SPINE": "./audio/non_neutral_spine.mp3"
        }
        self.last_prediction = None
        self.error_cooldown = {}  # To prevent audio spam
        self.cooldown_duration = 3.0  # seconds between same error audio
        
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
    
    def should_play_audio(self, error_type):
        """Check if enough time has passed since last audio for this error type"""
        import time
        current_time = time.time()
        
        if error_type not in self.error_cooldown:
            self.error_cooldown[error_type] = 0
            
        if current_time - self.error_cooldown[error_type] > self.cooldown_duration:
            self.error_cooldown[error_type] = current_time
            return True
        return False
    
    def check_and_play_feedback(self, prediction, confidence):
        """Check prediction and play appropriate audio feedback - only for incorrect form"""
        if not self.audio_enabled or not prediction or confidence < 0.7:
            return
            
        # Map predictions to audio files and check for changes
        prediction_lower = prediction.lower()
        
        # Only play audio for incorrect form - no audio for correct posture
        if "grip_narrow" in prediction_lower or "narrow" in prediction_lower:
            if self.should_play_audio("BAR_GRIP_NARROW"):
                self.play_audio(self.audio_files["BAR_GRIP_NARROW"])
                
        elif "grip_wide" in prediction_lower or "wide" in prediction_lower:
            if self.should_play_audio("BAR_GRIP_WIDE"):
                self.play_audio(self.audio_files["BAR_GRIP_WIDE"])
                
        elif "spine" in prediction_lower or "back" in prediction_lower or "neutral" in prediction_lower:
            if self.should_play_audio("NON_NEUTRAL_SPINE"):
                self.play_audio(self.audio_files["NON_NEUTRAL_SPINE"])
        
        # Update last prediction
        self.last_prediction = prediction
    
    def play_rep_completion_audio(self):
        """Play audio when a rep is completed correctly - removed to avoid audio for correct form"""
        # No audio for correct rep completion to keep it silent for good form
        pass

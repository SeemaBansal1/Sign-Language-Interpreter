"""
Flask Web Application for Real-Time Sign Language Translation
==============================================================
This application streams webcam video with real-time sign language recognition
and text-to-speech output.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.

Features:
    - Real-time hand landmark detection
    - Sign language prediction overlay
    - Text-to-speech when sign is held for 1+ second
    - Web-based interface
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
import os
from flask import Flask, render_template, Response, jsonify

# Text-to-speech (runs in separate thread)
import pyttsx3


# ============================================================================
# Flask Application Setup
# ============================================================================

app = Flask(__name__)

# ============================================================================
# Text-to-Speech Manager (Thread-safe)
# ============================================================================

class TTSManager:
    """Thread-safe text-to-speech manager."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.is_speaking = False
        self.speak_start_time = 0
        self.speak_timeout = 5.0  # Max seconds to wait before allowing new speech
    
    def speak(self, text):
        """Speak text in a separate thread."""
        current_time = time.time()
        
        # Reset is_speaking if it's been too long (timeout protection)
        if self.is_speaking and (current_time - self.speak_start_time) > self.speak_timeout:
            self.is_speaking = False
        
        if self.is_speaking:
            return
        
        def _speak():
            self.is_speaking = True
            self.speak_start_time = time.time()
            try:
                # Create a NEW engine instance in this thread (required for Windows)
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Try to set a good voice
                voices = engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
    
    def test_speech(self):
        """Test if TTS is working."""
        print("🔊 Testing text-to-speech...")
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say("Ready")
            engine.runAndWait()
            engine.stop()
            print("✅ TTS ready")
            return True
        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return False


# Initialize TTS manager
tts_manager = TTSManager()

# ============================================================================
# Sign Language Recognizer
# ============================================================================

class SignLanguageRecognizer:
    """Real-time sign language recognition using MediaPipe and trained model."""
    
    def __init__(self, model_path="model.p"):
        """Initialize the recognizer."""
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load trained model
        self.model = None
        self.labels = []
        self._load_model(model_path)
        
        # State for stable prediction and audio
        self.current_prediction = None
        self.prediction_start_time = None
        self.last_spoken_prediction = None
        self.hold_threshold = 1.0  # Seconds to hold before speaking
        
        # State for display
        self.display_prediction = ""
        self.prediction_confidence = 0.0
        self.prediction_held_time = 0.0
    
    def _load_model(self, model_path):
        """Load the trained model."""
        if not os.path.exists(model_path):
            print(f"Warning: Model file '{model_path}' not found!")
            print("Please run train_model.py first.")
            return
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.labels = list(model_data['labels'])
            print(f"Model loaded successfully!")
            print(f"Classes: {self.labels}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _extract_landmarks(self, hand_landmarks):
        """Extract normalized landmark coordinates."""
        data = []
        for landmark in hand_landmarks.landmark:
            data.extend([landmark.x, landmark.y, landmark.z])
        return np.array(data).reshape(1, -1)
    
    def _update_prediction_state(self, prediction, confidence):
        """Update prediction state and trigger audio if held long enough."""
        current_time = time.time()
        
        if prediction != self.current_prediction:
            # New prediction - reset timer
            self.current_prediction = prediction
            self.prediction_start_time = current_time
            self.prediction_held_time = 0
            # Reset last_spoken when prediction changes so new gesture can speak
            if self.last_spoken_prediction != prediction:
                self.last_spoken_prediction = None  # Allow new prediction to speak
        else:
            # Same prediction - update held time
            if self.prediction_start_time:
                self.prediction_held_time = current_time - self.prediction_start_time
                
                # Check if should speak - speak when held for 1+ second and hasn't spoken this gesture yet
                if (self.prediction_held_time >= self.hold_threshold and 
                    self.last_spoken_prediction != prediction and
                    confidence > 0.5):
                    
                    # Trigger TTS
                    tts_manager.speak(prediction)
                    self.last_spoken_prediction = prediction
                    print(f"🔊 Speaking: {prediction}")
        
        self.display_prediction = prediction
        self.prediction_confidence = confidence
    
    def _draw_overlay(self, frame, hand_detected):
        """Draw information overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Top overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Sign Language Translator", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Prediction display
        if self.display_prediction:
            # Prediction text
            pred_text = f"Sign: {self.display_prediction}"
            cv2.putText(frame, pred_text, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Confidence bar
            conf_width = int(200 * self.prediction_confidence)
            cv2.rectangle(frame, (250, 55), (450, 75), (50, 50, 50), -1)
            cv2.rectangle(frame, (250, 55), (250 + conf_width, 75), (0, 255, 0), -1)
            cv2.putText(frame, f"{self.prediction_confidence*100:.0f}%", (460, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Hold progress bar (shows progress to audio)
            if self.prediction_held_time > 0:
                progress = min(self.prediction_held_time / self.hold_threshold, 1.0)
                prog_width = int(200 * progress)
                cv2.rectangle(frame, (250, 80), (450, 95), (50, 50, 50), -1)
                
                # Color changes from yellow to green when ready
                color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
                cv2.rectangle(frame, (250, 80), (250 + prog_width, 95), color, -1)
                
                status = "Speaking..." if progress >= 1.0 else f"Hold: {self.prediction_held_time:.1f}s"
                cv2.putText(frame, status, (460, 92),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            status = "Show a hand sign..." if hand_detected else "No hand detected"
            cv2.putText(frame, status, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame for sign language recognition."""
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with custom style
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Make prediction if model is loaded
                if self.model is not None:
                    try:
                        features = self._extract_landmarks(hand_landmarks)
                        
                        # Get prediction probabilities
                        proba = self.model.predict_proba(features)[0]
                        pred_idx = np.argmax(proba)
                        confidence = proba[pred_idx]
                        prediction = self.labels[pred_idx]
                        
                        # Update state
                        self._update_prediction_state(prediction, confidence)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
        else:
            # No hand detected - reset ALL state including last spoken
            # This allows the same sign to be spoken again after removing hand
            if self.current_prediction is not None:
                self.current_prediction = None
                self.prediction_start_time = None
                self.prediction_held_time = 0
                self.display_prediction = ""
                self.last_spoken_prediction = None  # Reset so same sign can speak again
        
        # Draw overlay
        frame = self._draw_overlay(frame, hand_detected)
        
        return frame


# ============================================================================
# Video Stream Generator
# ============================================================================

# Global recognizer instance
recognizer = None

def init_recognizer():
    """Initialize the recognizer on first use."""
    global recognizer
    if recognizer is None:
        recognizer = SignLanguageRecognizer()


def generate_frames():
    """Generator function for video streaming."""
    init_recognizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Video stream started")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Process frame
            processed_frame = recognizer.process_frame(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            # Yield frame in multipart format
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        cap.release()
        print("Video stream stopped")


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def home():
    """Serve the home/landing page."""
    return render_template('home.html')


@app.route('/translator')
def translator():
    """Serve the translator page."""
    return render_template('index.html')


@app.route('/guide')
def guide():
    """Serve the ASL alphabet guide page."""
    return render_template('guide.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """Get current recognition status."""
    init_recognizer()
    return jsonify({
        'prediction': recognizer.display_prediction if recognizer else '',
        'confidence': recognizer.prediction_confidence if recognizer else 0,
        'held_time': recognizer.prediction_held_time if recognizer else 0,
        'model_loaded': recognizer.model is not None if recognizer else False,
        'labels': recognizer.labels if recognizer else []
    })


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR - WEB APPLICATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('model.p'):
        print("\nWarning: No trained model found!")
        print("The application will start, but prediction won't work.")
        print("Please run these steps first:")
        print("  1. python collect_data.py  (collect training data)")
        print("  2. python train_model.py   (train the model)")
    
    # Test TTS on startup
    print("\nTesting text-to-speech...")
    tts_manager.test_speech()
    
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False to avoid double-loading issues
        threaded=True
    )


if __name__ == '__main__':
    main()

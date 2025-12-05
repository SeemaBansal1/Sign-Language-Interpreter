# 🤟 Real-Time Sign Language Translator

A complete AI-powered application that translates sign language gestures to text and speech in real-time using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-red)

## ✨ Features

- **Real-time hand tracking** using MediaPipe (21 landmarks)
- **Machine Learning classification** with Random Forest
- **Text-to-Speech output** when sign is held for 1+ second
- **Web-based interface** with live video feed
- **Easy data collection** for training custom signs
- **Modern UI** with confidence bars and progress indicators

## 📁 Project Structure

```
sign-language-translator/
├── collect_data.py      # Collect training data
├── train_model.py       # Train the ML model
├── app.py               # Flask web application
├── requirements.txt     # Python dependencies
├── data.csv             # Collected landmark data (generated)
├── model.p              # Trained model (generated)
├── static/
│   └── style.css        # Stylesheet
└── templates/
    └── index.html       # Web interface
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SeemaBansal1/Sign-Language-Interpreter.git
cd Sign-Language-Interpreter
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Collect Training Data

```bash
python collect_data.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `S` | Save current hand pose |
| `C` | Change sign label |
| `Q` | Quit and save data |

> 💡 **Tip:** Collect at least 50 samples per sign for better accuracy.

### 5. Train the Model

```bash
python train_model.py
```

This creates `model.p` with your trained classifier.

### 6. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## 🎮 How to Use

1. **Position your hand** clearly in front of the camera
2. **Make a sign gesture** from your trained set
3. **Hold the sign for 1 second** to trigger audio output
4. **Remove your hand briefly** before making the next sign

## 📊 Training Tips

| Aspect | Recommendation |
|--------|----------------|
| Samples per sign | 50-100 minimum |
| Lighting | Consistent, well-lit |
| Background | Plain, non-distracting |
| Hand position | Vary slightly for robustness |
| Distance | Keep hand at consistent distance |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Hand Detection | MediaPipe Hands |
| ML Model | Scikit-learn Random Forest |
| Backend | Flask |
| Video Processing | OpenCV |
| Text-to-Speech | pyttsx3 |
| Frontend | HTML5, CSS3, JavaScript |

## 📋 Requirements

- **Python 3.11** (recommended, tested)
- Webcam
- Windows/macOS/Linux

## 🔧 Configuration

### Adjust Hold Time for Speech

In `app.py`, modify line 124:

```python
self.hold_threshold = 1.0  # Seconds (change to 0.5 for faster, 2.0 for slower)
```

### Adjust Detection Confidence

In `app.py`, modify lines 111-112:

```python
min_detection_confidence=0.7,  # Lower = more sensitive
min_tracking_confidence=0.5
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/status` | GET | Current prediction status (JSON) |

### Status Response Example

```json
{
  "prediction": "Hello",
  "confidence": 0.95,
  "held_time": 1.2,
  "model_loaded": true,
  "labels": ["Hello", "Yes", "No"]
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Webcam not detected | Check camera permissions, try different index in `cv2.VideoCapture(0)` |
| Low accuracy | Collect more training data, ensure consistent lighting |
| Audio not working | Check system volume, verify pyttsx3 installation |
| Slow performance | Close other applications, reduce camera resolution |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-sign`)
3. Commit changes (`git commit -am 'Add new sign support'`)
4. Push to branch (`git push origin feature/new-sign`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [Scikit-learn](https://scikit-learn.org/) for machine learning
- [Flask](https://flask.palletsprojects.com/) for web framework
- [OpenCV](https://opencv.org/) for computer vision

---


<p align="center">
  Made with ❤️ for accessibility
</p>

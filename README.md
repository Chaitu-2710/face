# Real-time Deepfake Detector for Video Calls

This project implements a real-time deepfake detection system specifically designed for video calls and online meetings. It uses computer vision and deep learning techniques to analyze video streams and identify potential deepfake manipulations.

## Features

- Real-time face detection and tracking
- Multiple deepfake detection methods:
  - Facial landmark consistency
  - Texture analysis
  - Blinking pattern analysis
  - Head pose estimation
  - Audio-visual sync detection
- Support for various video sources (webcam, video files, screen capture)
- User-friendly GUI with detection confidence scores
- Alert system for suspicious activity

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main application:
```bash
python deepfake_detector_app.py
```

## Usage

1. Launch the application
2. Select the video source (webcam/video file/screen capture)
3. The detector will start analyzing the video stream in real-time
4. Detection results and confidence scores will be displayed on screen
5. Press 'q' to quit the application

## Detection Methods

The system uses multiple detection methods to identify potential deepfakes:

1. **Facial Landmark Analysis**: Tracks facial landmarks to detect unnatural movements
2. **Texture Analysis**: Analyzes skin texture for artificial patterns
3. **Blink Detection**: Monitors natural blinking patterns
4. **Head Pose Estimation**: Checks for consistent head movements
5. **Audio-Visual Sync**: Verifies lip sync in video calls

# Real-Time Webcam Object Detection (YOLO + OpenCV)

A real-time computer vision project that detects objects from a live webcam feed using YOLO (Ultralytics) and OpenCV.

## Features
- Live webcam object detection
- FPS + inference time overlay
- Config-driven thresholds and class filtering (YAML)
- Alert banner for chosen objects (e.g., "cell phone")
- Optional video recording of detections

## Setup (Windows)
```powershell
uv venv
.venv\Scripts\activate
uv add ultralytics opencv-python pyyaml numpy

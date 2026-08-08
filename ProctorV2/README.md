# Classroom Proctor System V2

A Flask + OpenCV face-recognition web app for classroom monitoring, student registration, attendance tracking, and unknown-face handling.

## Features

- Live webcam monitoring
- Student registration with auto-capture
- Attendance logging and session tracking
- Unknown face saving and retraining flow
- Web UI for managing students, unknown faces, and sessions

## Requirements

- Python 3.9+
- OpenCV with contrib (`opencv-contrib-python`)
- Flask and Flask-CORS

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows PowerShell:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   python app.py
   ```

4. Open http://localhost:5000

## Notes

- The app uses the local camera by default. If your camera is not detected, try changing the camera index in `app.py` from `0` to `1`.
- The `data/` folder stores students, unknown faces, sessions, and the trained model.

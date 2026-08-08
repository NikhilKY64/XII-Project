# Classroom Proctor System V2 — Setup Guide

## What's New in V2
- Modern web UI (HTML/CSS) — runs in your browser
- Session management — start/stop class sessions
- Attendance tracking with CSV export
- Detection cooldown (no duplicate logs)
- Student profile photos
- Unknown face thumbnails with register shortcut
- Settings page — tune all parameters in UI
- Toast notifications instead of popups

---

## File Structure

```
app.py                  ← Flask server + all API routes
face_db.py              ← Face recognition + attendance logic
index.html              ← Full web UI (single page)
setup.md                ← This file

data/                   ← auto-created
  students.json
  sessions.json
  face_model.pkl(.yml)
  students/<id>/
  unknown/
```

---

## Installation

### 1. Virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Use `opencv-contrib-python`, NOT `opencv-python`.

---

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Run locally (no Docker)

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

> If your camera does not open, try changing the camera index in `app.py` from `0` to `1`.

---

## Usage

### 📡 Live Monitor
- Real-time camera feed with face detection overlays
- **Green box** = recognized student, **Red box** = unknown
- Start/Stop sessions from the panel on the right
- Detection log shows who was seen and when

### ➕ Register Student
- Enter ID + Name, click **Start Capture**
- App auto-captures 10 photos while you look at camera
- Model retrains automatically after saving

### 👥 Students
- Card grid of all registered students with photos
- Hover to reveal delete button

### ❓ Unknown Faces
- Thumbnails of unrecognized faces from monitoring
- Click **+ Register** to assign to a student
- Or **✕** to delete

### 📋 Sessions
- Lists all past and active sessions
- Click any session to see full attendance table
- Export attendance as CSV

### ⚙️ Settings
- Confidence Threshold (40–100, lower = stricter)
- Attendance Cooldown (seconds between re-logging same student)
- Capture Count (photos per registration)
- Unknown Save Cooldown (delay between saving unknown faces)

---

## Tips

| Setting | Recommendation |
|---|---|
| Good lighting | Even, bright lighting improves accuracy significantly |
| Register angles | Capture slightly left/right too for better real-world accuracy |
| Confidence | Start at 70, lower to 60 if false negatives are common |
| Cooldown | 30s is good for class — prevents log spam |

---

## Troubleshooting

**`AttributeError: module 'cv2' has no attribute 'face'`**
```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

**`ModuleNotFoundError: No module named 'flask_cors'`**
```bash
pip install flask-cors
```

**Camera not working**
Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py`

**Port already in use**
Change `port=5000` to `port=5001` at the bottom of `app.py`

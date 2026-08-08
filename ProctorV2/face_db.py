"""
face_db.py - Face database, recognition, attendance logic
"""

import os
import shutil
import urllib.request
import cv2
import json
import pickle
import numpy as np
from datetime import datetime

DATA_DIR = "data"
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown")
CASCADE_DIR = os.path.join(DATA_DIR, "cascades")
DB_FILE = os.path.join(DATA_DIR, "students.json")
MODEL_FILE = os.path.join(DATA_DIR, "face_model.pkl")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    + CASCADE_FILE
)

for d in [DATA_DIR, STUDENTS_DIR, UNKNOWN_DIR, CASCADE_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIDENCE_THRESHOLD = 80
UNKNOWN_SAVING_ENABLED = True


# ─── Student Database ────────────────────────────────────────────────────────

def load_db() -> dict:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db: dict):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def get_student_dir(student_id: str) -> str:
    path = os.path.join(STUDENTS_DIR, student_id)
    os.makedirs(path, exist_ok=True)
    return path

def register_student(student_id: str, name: str, images: list) -> bool:
    db = load_db()
    db[student_id] = {"name": name, "registered": datetime.now().isoformat()}
    save_db(db)
    sdir = get_student_dir(student_id)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(sdir, f"{i}.jpg"), img)
    train_model()
    return True


def add_images_to_student(student_id: str, images: list) -> bool:
    db = load_db()
    if student_id not in db:
        return False
    sdir = get_student_dir(student_id)
    existing_files = [f for f in os.listdir(sdir) if f.endswith(".jpg")]
    next_index = max([int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()], default=-1) + 1
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(sdir, f"{next_index + i}.jpg"), img)
    train_model()
    return True

def delete_student(student_id: str) -> bool:
    db = load_db()
    if student_id not in db:
        return False
    del db[student_id]
    save_db(db)
    import shutil
    sdir = os.path.join(STUDENTS_DIR, student_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    train_model()
    return True

def get_student_photo_path(student_id: str) -> str | None:
    sdir = os.path.join(STUDENTS_DIR, student_id)
    if not os.path.exists(sdir):
        return None
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".jpg"):
            return os.path.join(sdir, f)
    return None


def add_unknown_images_to_student(student_id: str, unknown_filenames: list) -> dict:
    db = load_db()
    if student_id not in db:
        return {"ok": False, "error": "Student not found"}

    sdir = get_student_dir(student_id)
    added_files = []
    skipped_files = []
    failed_files = []

    for filename in unknown_filenames:
        if not isinstance(filename, str) or not filename:
            continue
        source_path = os.path.join(UNKNOWN_DIR, filename)
        if not os.path.exists(source_path):
            failed_files.append(filename)
            continue

        try:
            img = cv2.imread(source_path)
            if img is None:
                failed_files.append(filename)
                continue

            stem, ext = os.path.splitext(os.path.basename(filename))
            target_name = f"{stem}{ext or '.jpg'}"
            target_path = os.path.join(sdir, target_name)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(sdir, f"{stem}_{counter}{ext or '.jpg'}")
                counter += 1

            shutil.move(source_path, target_path)
            added_files.append(filename)
        except Exception as exc:
            failed_files.append(filename)

    if added_files:
        train_model()

    return {
        "ok": True,
        "added": len(added_files),
        "skipped": len(skipped_files),
        "failed": failed_files,
        "skipped_files": skipped_files,
        "added_files": added_files,
        "trained": len(added_files) > 0,
    }


# ─── Model Training ──────────────────────────────────────────────────────────

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    db = load_db()
    faces, labels, label_map = [], [], {}
    face_cascade = _get_cascade()

    for idx, (student_id, info) in enumerate(db.items()):
        label_map[idx] = {"id": student_id, "name": info["name"]}
        sdir = os.path.join(STUDENTS_DIR, student_id)
        if not os.path.exists(sdir):
            continue
        for fname in os.listdir(sdir):
            if not fname.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(sdir, fname))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(detected) == 0:
                equalized = cv2.equalizeHist(gray)
                detected = face_cascade.detectMultiScale(equalized, 1.05, 4, minSize=(40, 40))

            if len(detected) == 0:
                # Last resort: use the full grayscale ROI if the image is likely a face crop.
                faces.append(gray)
                labels.append(idx)
                continue

            for (x, y, w, h) in detected:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(idx)

    if not faces:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"trained": False, "label_map": {}}, f)
        return

    recognizer.train(faces, np.array(labels))
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"trained": True, "label_map": label_map}, f)
    recognizer.save(MODEL_FILE + ".yml")

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, {}
    with open(MODEL_FILE, "rb") as f:
        meta = pickle.load(f)
    if not meta.get("trained"):
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE + ".yml")
    return recognizer, meta["label_map"]


# ─── Recognition ─────────────────────────────────────────────────────────────

def recognize_face(gray_face, recognizer, label_map):
    if recognizer is None:
        return "unknown", "Unknown", 0.0
    label, confidence = recognizer.predict(gray_face)
    if confidence < CONFIDENCE_THRESHOLD:
        info = label_map.get(label, {})
        return info.get("id", "unknown"), info.get("name", "Unknown"), confidence
    return "unknown", "Unknown", confidence


# ─── Face Detection ───────────────────────────────────────────────────────────

def detect_faces(frame):
    cascade = _get_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # If no faces were found on the first pass, try a more sensitive fallback.
    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40)
        )

    return gray, list(faces)

def _get_cascade():
    if getattr(_get_cascade, "_cascade", None) is not None:
        return _get_cascade._cascade

    candidates = [
        os.path.join(cv2.data.haarcascades, CASCADE_FILE),
        os.path.join(os.path.dirname(cv2.__file__), "data", CASCADE_FILE),
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascades", CASCADE_FILE),
        os.path.join(CASCADE_DIR, CASCADE_FILE),
    ]
    cascade_path = next((p for p in candidates if os.path.exists(p)), None)

    if cascade_path is None:
        try:
            urllib.request.urlretrieve(CASCADE_URL, os.path.join(CASCADE_DIR, CASCADE_FILE))
            cascade_path = os.path.join(CASCADE_DIR, CASCADE_FILE)
            print(f"Downloaded Haar cascade to {cascade_path}")
        except Exception as exc:
            raise FileNotFoundError(
                "Haar cascade file not found and download failed. "
                f"Tried: {candidates}. Error: {exc}"
            ) from exc

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade classifier from {cascade_path}")

    _get_cascade._cascade = cascade
    return cascade


# ─── Unknown Faces ───────────────────────────────────────────────────────────

def set_unknown_saving_enabled(enabled: bool):
    global UNKNOWN_SAVING_ENABLED
    UNKNOWN_SAVING_ENABLED = bool(enabled)


def save_unknown_face(face_img) -> str | None:
    if not UNKNOWN_SAVING_ENABLED:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(path, face_img)
    return path

def list_unknown_faces() -> list:
    if not os.path.exists(UNKNOWN_DIR):
        return []
    return sorted([
        f for f in os.listdir(UNKNOWN_DIR) if f.endswith(".jpg")
    ], reverse=True)

def promote_unknown_to_student(filename: str, student_id: str, name: str) -> bool:
    path = os.path.join(UNKNOWN_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        return False
    ok = register_student(student_id, name, [img])
    if ok:
        os.remove(path)
    return ok

def delete_unknown_face(filename: str) -> bool:
    path = os.path.join(UNKNOWN_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def clear_unknown_faces():
    for f in list_unknown_faces():
        os.remove(os.path.join(UNKNOWN_DIR, f))


# ─── Sessions & Attendance ───────────────────────────────────────────────────

def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {"active": None, "sessions": {}}

def save_sessions(data: dict):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def start_session(name: str) -> str:
    data = load_sessions()
    sid = datetime.now().strftime("SES_%Y%m%d_%H%M%S")
    data["active"] = sid
    data["sessions"][sid] = {
        "name": name,
        "started": datetime.now().isoformat(),
        "ended": None,
        "attendance": {}
    }
    save_sessions(data)
    return sid

def end_session() -> str | None:
    data = load_sessions()
    sid = data.get("active")
    if not sid:
        return None
    data["sessions"][sid]["ended"] = datetime.now().isoformat()
    data["active"] = None
    save_sessions(data)
    return sid

def get_active_session() -> tuple:
    data = load_sessions()
    sid = data.get("active")
    if not sid:
        return None, None
    return sid, data["sessions"].get(sid)

def mark_attendance(student_id: str, student_name: str):
    data = load_sessions()
    sid = data.get("active")
    if not sid:
        return
    sess = data["sessions"][sid]
    if student_id not in sess["attendance"]:
        sess["attendance"][student_id] = {
            "name": student_name,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "count": 1
        }
    else:
        sess["attendance"][student_id]["last_seen"] = datetime.now().isoformat()
        sess["attendance"][student_id]["count"] += 1
    save_sessions(data)

def get_all_sessions() -> list:
    data = load_sessions()
    result = []
    db = load_db()
    for sid, sess in data["sessions"].items():
        total = len(db)
        present = len(sess["attendance"])
        result.append({
            "id": sid,
            "name": sess["name"],
            "started": sess["started"],
            "ended": sess["ended"],
            "present": present,
            "total": total,
            "active": data["active"] == sid
        })
    return sorted(result, key=lambda x: x["started"], reverse=True)

def get_session_detail(session_id: str) -> dict | None:
    data = load_sessions()
    sess = data["sessions"].get(session_id)
    if not sess:
        return None
    db = load_db()
    all_students = []
    for sid, info in db.items():
        att = sess["attendance"].get(sid)
        all_students.append({
            "id": sid,
            "name": info["name"],
            "present": att is not None,
            "first_seen": att["first_seen"] if att else None,
            "count": att["count"] if att else 0
        })
    return {
        "id": session_id,
        "name": sess["name"],
        "started": sess["started"],
        "ended": sess["ended"],
        "students": all_students
    }

def export_session_csv(session_id: str) -> str:
    detail = get_session_detail(session_id)
    if not detail:
        return ""
    lines = ["Student ID,Name,Present,First Seen,Detection Count"]
    for s in detail["students"]:
        lines.append(f"{s['id']},{s['name']},{'Yes' if s['present'] else 'No'},{s['first_seen'] or ''},{ s['count']}")
    return "\n".join(lines)

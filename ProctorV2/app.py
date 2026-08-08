"""
app.py - Classroom Proctor System V2
Run: python app.py  →  open http://localhost:5000
"""

import cv2
import json
import base64
import threading
import time
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import io
import os

import face_db

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ─── Camera State ─────────────────────────────────────────────────────────────

class CameraState:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.running = False
        self.face_count = 0
        self.recent_detections = []   # [{id, name, time, known}]
        self.recognizer = None
        self.label_map = {}
        self.last_seen = {}           # student_id -> timestamp (cooldown)
        self.unknown_cooldown = {}    # hash -> timestamp
        self.unknown_saving_enabled = True
        self.capture_mode = False
        self.capture_target = 10
        self.capture_mode_type = "new"
        self.auto_detect_enabled = False
        self.detected_student = None
        self.roi = None
        self.captured_images = []
        self.capture_interval_sec = 0.5
        self.capture_stability_sec = 0.6
        self.last_capture_time = 0.0
        self.capture_stable_since = None
        self.last_capture_box = None
        self.COOLDOWN_SEC = 30
        self.UNKNOWN_COOLDOWN_SEC = 10
        self.trackers = []
        self.tracked_faces = []
        self.tracker_frame_count = 0
        self.DETECTION_INTERVAL = 8
        self.RECOGNITION_INTERVAL = 3.0
        self.BBOX_SMOOTHING = 0.75

    def reload_model(self):
        self.recognizer, self.label_map = face_db.load_model()

cam = CameraState()


def create_tracker():
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    raise RuntimeError("No supported OpenCV tracker found. Install opencv-contrib-python.")


def init_trackers(frame, gray, faces):
    cam.trackers = []
    cam.tracked_faces = []
    for (x, y, w, h) in faces:
        try:
            tracker = create_tracker()
            tracker.init(frame, (x, y, w, h))
        except Exception:
            continue
        face_gray = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        sid, name, conf = face_db.recognize_face(face_gray, cam.recognizer, cam.label_map)
        cam.tracked_faces.append({
            "tracker": tracker,
            "box": (x, y, w, h),
            "sid": sid,
            "name": name,
            "conf": conf,
            "known": sid != "unknown",
            "last_seen": time.time(),
        })
    cam.tracker_frame_count = 0


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def update_trackers(frame):
    updated = []
    for face in cam.tracked_faces:
        ok, box = face["tracker"].update(frame)
        if not ok:
            continue
        x, y, w, h = [int(v) for v in box]
        px, py, pw, ph = face["box"]
        alpha = cam.BBOX_SMOOTHING
        x = int(px * alpha + x * (1 - alpha))
        y = int(py * alpha + y * (1 - alpha))
        w = int(pw * alpha + w * (1 - alpha))
        h = int(ph * alpha + h * (1 - alpha))
        if w <= 0 or h <= 0:
            continue
        # Clamp to frame dimensions
        h_max, w_max = frame.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = max(1, min(w, w_max - x))
        h = max(1, min(h, h_max - y))
        face["box"] = (x, y, w, h)
        updated.append(face)
    cam.tracked_faces = updated
    cam.tracker_frame_count += 1
    return len(updated) > 0


def match_detections(faces, frame, gray):
    new_tracked = []
    matched = [False] * len(cam.tracked_faces)

    for det in faces:
        best_idx = None
        best_iou = 0.0
        for idx, face in enumerate(cam.tracked_faces):
            iou = bbox_iou(det, face["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou > 0.3:
            face = cam.tracked_faces[best_idx]
            face["box"] = det
            face["missed"] = 0
            try:
                tracker = create_tracker()
                tracker.init(frame, tuple(det))
                face["tracker"] = tracker
            except Exception:
                pass
            x, y, w, h = det

            # Only recognize periodically instead of every detection
            now = time.time()

            if now - face.get("recognition_time", 0) >= cam.RECOGNITION_INTERVAL:
                face_gray = cv2.resize(
                    gray[y:y+h, x:x+w],
                    (100, 100)
                )

                sid, name, conf = face_db.recognize_face(
                    face_gray,
                    cam.recognizer,
                    cam.label_map
                )

                face.update({
                    "sid": sid,
                    "name": name,
                    "conf": conf,
                    "known": sid != "unknown",
                    "recognition_time": now
                })

            face["last_seen"] = now
        else:
            try:
                tracker = create_tracker()
                tracker.init(frame, tuple(det))
            except Exception:
                continue
            x, y, w, h = det

            face_gray = cv2.resize(
                gray[y:y+h, x:x+w],
                (100, 100)
            )

            sid, name, conf = face_db.recognize_face(
                face_gray,
                cam.recognizer,
                cam.label_map
            )

            new_tracked.append({
                "tracker": tracker,
                "box": det,
                "sid": sid,
                "name": name,
                "conf": conf,
                "known": sid != "unknown",
                "last_seen": time.time(),
                "recognition_time": time.time(),
                "missed": 0,
            })

    for idx, face in enumerate(cam.tracked_faces):
        if not matched[idx]:
            face["missed"] += 1

            if face["missed"] <= 4:
                new_tracked.append(face)

    cam.tracked_faces = new_tracked
    cam.tracker_frame_count = 0


def reset_trackers():
    cam.trackers = []
    cam.tracked_faces = []
    cam.tracker_frame_count = 0


def filter_faces_in_roi(faces, frame, roi):
    if not roi:
        return list(faces)
    rx = int(roi["x"] * frame.shape[1])
    ry = int(roi["y"] * frame.shape[0])
    rw = int(roi["w"] * frame.shape[1])
    rh = int(roi["h"] * frame.shape[0])
    rx2 = min(frame.shape[1], rx + rw)
    ry2 = min(frame.shape[0], ry + rh)
    filtered = []
    for (x, y, w, h) in faces:
        cx = x + (w / 2)
        cy = y + (h / 2)
        if rx <= cx <= rx2 and ry <= cy <= ry2:
            filtered.append((x, y, w, h))
    return filtered


def camera_thread():
    cam.cap = cv2.VideoCapture(1) 
    cam.running = True

    while cam.running:
        ret, frame = cam.cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        # Resize very large video frames for faster processing
        h, w = frame.shape[:2]

        if w > 960:
            scale = 960 / w
            frame = cv2.resize(
                frame,
                (960, int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        with cam.lock:
            cam.latest_frame = frame.copy()

        display = frame.copy()
        if cam.capture_mode:
            try:
                gray, faces = face_db.detect_faces(frame)
            except Exception as exc:
                print(f"Face detection failed: {exc}")
                faces = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_faces = filter_faces_in_roi(faces, frame, cam.roi)
            cam.face_count = len(roi_faces)
            cam.detected_student = None
            if cam.capture_mode_type == "existing" and cam.auto_detect_enabled and len(roi_faces) == 1:
                (x, y, w, h) = roi_faces[0]
                face_gray = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                sid, name, conf = face_db.recognize_face(face_gray, cam.recognizer, cam.label_map)
                if sid != "unknown" and conf < face_db.CONFIDENCE_THRESHOLD:
                    cam.detected_student = {"id": sid, "name": name, "confidence": round(conf, 1)}
                    cv2.putText(display, f"Detected {name}", (x, max(0, y-28)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 100), 1)
            now_time = time.time()
            if len(roi_faces) == 1:
                (x, y, w, h) = roi_faces[0]
                current_box = (x, y, w, h)
                rx = x
                ry = y
                rw = w
                rh = h
                if cam.roi:
                    rw = int(cam.roi["w"] * frame.shape[1])
                    rh = int(cam.roi["h"] * frame.shape[0])
                    rx = int(cam.roi["x"] * frame.shape[1])
                    ry = int(cam.roi["y"] * frame.shape[0])
                    current_box = (rx, ry, rw, rh)
                if cam.last_capture_box is None:
                    cam.last_capture_box = current_box
                    cam.capture_stable_since = now_time
                else:
                    same_face = abs(current_box[0] - cam.last_capture_box[0]) < 20 and abs(current_box[1] - cam.last_capture_box[1]) < 20 and abs(current_box[2] - cam.last_capture_box[2]) < 20 and abs(current_box[3] - cam.last_capture_box[3]) < 20
                    if same_face:
                        if cam.capture_stable_since is None:
                            cam.capture_stable_since = now_time
                        elif now_time - cam.capture_stable_since >= cam.capture_stability_sec:
                            if now_time - cam.last_capture_time >= cam.capture_interval_sec:
                                roi_frame = frame[max(0, ry):min(frame.shape[0], ry+rh), max(0, rx):min(frame.shape[1], rx+rw)]
                                if roi_frame.size > 0:
                                    cam.captured_images.append(roi_frame.copy())
                                else:
                                    cam.captured_images.append(frame.copy())
                                cam.last_capture_time = now_time
                                cam.last_capture_box = current_box
                                cam.capture_stable_since = now_time
                    else:
                        cam.last_capture_box = current_box
                        cam.capture_stable_since = now_time
            else:
                cam.last_capture_box = None
                cam.capture_stable_since = None

            for (x, y, w, h) in roi_faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 180, 0), 2)
                cv2.putText(display, f"Capturing {len(cam.captured_images)}/{cam.capture_target}",
                            (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        else:
            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            use_detection = (cam.tracker_frame_count % cam.DETECTION_INTERVAL == 0) or not cam.tracked_faces
            faces = []
            if use_detection:
                try:
                    gray, faces = face_db.detect_faces(frame)
                except Exception as exc:
                    print(f"Face detection failed: {exc}")
                    faces = []
            if faces:
                match_detections(faces, frame, gray)
            else:
                if cam.tracked_faces:
                    tracker_ok = update_trackers(frame)
                    if not tracker_ok:
                        reset_trackers()
                else:
                    cam.tracker_frame_count += 1

            cam.face_count = len(cam.tracked_faces)
            for face in cam.tracked_faces:
                (x, y, w, h) = face["box"]
                sid = face["sid"]
                name = face["name"]
                conf = face["conf"]
                if sid != "unknown":
                    color = (80, 220, 100)
                    label = f"{name} | {conf:.0f}%"
                    if now - cam.last_seen.get(sid, 0) > cam.COOLDOWN_SEC:
                        cam.last_seen[sid] = now
                        face_db.mark_attendance(sid, name)
                        det = {"id": sid, "name": name, "conf": round(conf, 1),
                               "time": datetime.now().strftime("%H:%M:%S"), "known": True}
                        cam.recent_detections.insert(0, det)
                        cam.recent_detections = cam.recent_detections[:50]
                else:
                    color = (60, 60, 220)
                    label = f"Unknown | {conf:.0f}%"
                    face_key = f"{x//40}_{y//40}"
                    if (cam.unknown_saving_enabled and not cam.capture_mode and
                        now - cam.unknown_cooldown.get(face_key, 0) > cam.UNKNOWN_COOLDOWN_SEC):
                        cam.unknown_cooldown[face_key] = now
                        face_db.save_unknown_face(frame[y:y+h, x:x+w])
                        det = {"id": "unknown", "name": "Unknown", "conf": round(conf, 1),
                               "time": datetime.now().strftime("%H:%M:%S"), "known": False}
                        cam.recent_detections.insert(0, det)
                        cam.recent_detections = cam.recent_detections[:50]

                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(display, (x, y-24), (x+w, y), color, -1)
                cv2.putText(display, label, (x+4, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        with cam.lock:
            cam.latest_annotated = display.copy()

        time.sleep(0.03)

    if cam.cap:
        cam.cap.release()


def gen_frames():
    while True:
        with cam.lock:
            frame = cam.latest_annotated if cam.latest_annotated is not None else cam.latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"), mimetype="text/html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    sid, sess = face_db.get_active_session()
    db = face_db.load_db()
    return jsonify({
        "face_count": cam.face_count,
        "student_count": len(db),
        "unknown_count": len(face_db.list_unknown_faces()),
        "unknown_saving_enabled": cam.unknown_saving_enabled,
        "model_trained": cam.recognizer is not None,
        "active_session": {"id": sid, "name": sess["name"] if sess else None} if sid else None,
        "recent_detections": cam.recent_detections[:10]
    })

# ── Students ──

@app.route("/api/students", methods=["GET"])
def api_students():
    db = face_db.load_db()
    result = []
    for sid, info in db.items():
        photo = face_db.get_student_photo_path(sid)
        result.append({
            "id": sid,
            "name": info["name"],
            "registered": info.get("registered", "")[:19].replace("T", " "),
            "has_photo": photo is not None
        })
    return jsonify(result)

@app.route("/api/students/<sid>/photo")
def student_photo(sid):
    path = face_db.get_student_photo_path(sid)
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404

@app.route("/api/students/<sid>", methods=["DELETE"])
def delete_student(sid):
    ok = face_db.delete_student(sid)
    if ok:
        cam.reload_model()
    return jsonify({"ok": ok})

# ── Registration ──

@app.route("/api/register/start", methods=["POST"])
def register_start():
    data = request.json
    sid = data.get("id", "").strip()
    name = data.get("name", "").strip()
    mode = data.get("mode", "new")
    target = int(data.get("target", 10))
    auto_detect = bool(data.get("auto_detect", False))
    roi = data.get("roi")
    if not sid or not name:
        return jsonify({"ok": False, "error": "Missing ID or name"})
    db = face_db.load_db()
    if mode == "new" and sid in db:
        return jsonify({"ok": False, "error": f"Student ID '{sid}' already exists"})
    if mode == "existing" and sid not in db:
        return jsonify({"ok": False, "error": f"Student ID '{sid}' not found"})
    cam.capture_mode = True
    cam.capture_mode_type = mode
    cam.auto_detect_enabled = auto_detect
    cam.capture_target = max(5, min(30, target))
    cam.roi = roi if isinstance(roi, dict) else None
    cam.captured_images = []
    cam.last_capture_time = 0.0
    cam.capture_stable_since = None
    cam.last_capture_box = None
    cam.detected_student = None
    return jsonify({"ok": True, "mode": mode, "target": cam.capture_target})

@app.route("/api/register/status")
def register_status():
    return jsonify({
        "capturing": cam.capture_mode,
        "captured": len(cam.captured_images),
        "target": cam.capture_target,
        "face_count": cam.face_count,
        "mode": cam.capture_mode_type,
        "auto_detect_enabled": cam.auto_detect_enabled,
        "detected_student": cam.detected_student
    })

@app.route("/api/register/finish", methods=["POST"])
def register_finish():
    data = request.json
    sid = data.get("id", "").strip()
    name = data.get("name", "").strip()
    if not cam.captured_images:
        cam.capture_mode = False
        return jsonify({"ok": False, "error": "No images captured"})
    cam.capture_mode = False
    images = cam.captured_images.copy()
    cam.captured_images = []
    if cam.capture_mode_type == "existing":
        ok = face_db.add_images_to_student(sid, images)
    else:
        ok = face_db.register_student(sid, name, images)
    if ok:
        cam.reload_model()
    return jsonify({"ok": ok, "mode": cam.capture_mode_type})

@app.route("/api/register/cancel", methods=["POST"])
def register_cancel():
    cam.capture_mode = False
    cam.capture_mode_type = "new"
    cam.auto_detect_enabled = False
    cam.detected_student = None
    cam.roi = None
    cam.captured_images = []
    cam.last_capture_time = 0.0
    cam.capture_stable_since = None
    cam.last_capture_box = None
    return jsonify({"ok": True})

@app.route("/api/register/search")
def register_search():
    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", "name").lower()
    if not query:
        return jsonify({"ok": True, "results": []})

    db = face_db.load_db()
    results = []
    query_lower = query.lower()
    for sid, info in db.items():
        name = info.get("name", "")
        if mode == "id":
            match = query_lower in sid.lower()
        else:
            match = query_lower in name.lower()
        if match:
            results.append({"id": sid, "name": name})

    results.sort(key=lambda item: item["name"].lower())
    return jsonify({"ok": True, "results": results[:10]})

@app.route("/api/register/auto_detect", methods=["POST"])
def register_auto_detect():
    with cam.lock:
        frame = cam.latest_frame
        roi = cam.roi

    if frame is None:
        return jsonify({"ok": False, "error": "Camera feed not ready"})

    try:
        gray, faces = face_db.detect_faces(frame)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Detection failed: {exc}"})

    roi_faces = filter_faces_in_roi(faces, frame, roi)
    if len(roi_faces) != 1:
        return jsonify({"ok": False, "error": "No single face detected in the selected area"})

    x, y, w, h = roi_faces[0]
    face_crop = frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        return jsonify({"ok": False, "error": "Detected face crop is empty"})

    face_gray = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), (100, 100))
    sid, name, conf = face_db.recognize_face(face_gray, cam.recognizer, cam.label_map)

    if sid == "unknown" or conf >= face_db.CONFIDENCE_THRESHOLD:
        return jsonify({"ok": True, "recognized": False, "message": "No known face matched yet"})

    _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_data = base64.b64encode(buf).decode("ascii")
    cam.detected_student = {"id": sid, "name": name, "confidence": round(conf, 1)}

    return jsonify({
        "ok": True,
        "recognized": True,
        "student": {"id": sid, "name": name, "confidence": round(conf, 1)},
        "image": f"data:image/jpeg;base64,{image_data}",
        "message": "Known face detected"
    })

# ── Unknown Faces ──

@app.route("/api/unknown/saving", methods=["GET", "POST"])
def api_unknown_saving():
    if request.method == "POST":
        data = request.json or {}
        enabled = data.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in {"1", "true", "yes", "on"}
        cam.unknown_saving_enabled = bool(enabled)
        face_db.set_unknown_saving_enabled(cam.unknown_saving_enabled)
        return jsonify({"ok": True, "enabled": cam.unknown_saving_enabled})
    return jsonify({"enabled": cam.unknown_saving_enabled})

@app.route("/api/unknown")
def api_unknown():
    files = face_db.list_unknown_faces()
    return jsonify(files)

@app.route("/api/unknown/train", methods=["POST"])
def api_unknown_train():
    data = request.json or {}
    student_id = str(data.get("student_id", "")).strip()
    filenames = data.get("filenames", [])
    if not student_id:
        return jsonify({"ok": False, "error": "Missing student ID"})
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"ok": False, "error": "No unknown images selected"})

    result = face_db.add_unknown_images_to_student(student_id, filenames)
    if result.get("ok"):
        cam.reload_model()
    return jsonify(result)

@app.route("/api/unknown/<filename>/image")
def unknown_image(filename):
    path = os.path.join(face_db.UNKNOWN_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404

@app.route("/api/unknown/<filename>/promote", methods=["POST"])
def promote_unknown(filename):
    data = request.json
    sid = data.get("id", "").strip()
    name = data.get("name", "").strip()
    ok = face_db.promote_unknown_to_student(filename, sid, name)
    if ok:
        cam.reload_model()
    return jsonify({"ok": ok})

@app.route("/api/unknown/<filename>", methods=["DELETE"])
def delete_unknown(filename):
    ok = face_db.delete_unknown_face(filename)
    return jsonify({"ok": ok})

@app.route("/api/unknown/clear", methods=["POST"])
def clear_unknown():
    face_db.clear_unknown_faces()
    return jsonify({"ok": True})

# ── Sessions / Attendance ──

@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    return jsonify(face_db.get_all_sessions())

@app.route("/api/sessions/start", methods=["POST"])
def session_start():
    name = request.json.get("name", f"Class {datetime.now().strftime('%b %d %H:%M')}")
    sid = face_db.start_session(name)
    return jsonify({"ok": True, "id": sid})

@app.route("/api/sessions/end", methods=["POST"])
def session_end():
    sid = face_db.end_session()
    return jsonify({"ok": True, "id": sid})

@app.route("/api/sessions/<sid>")
def session_detail(sid):
    detail = face_db.get_session_detail(sid)
    if not detail:
        return jsonify({"error": "Not found"}), 404
    return jsonify(detail)

@app.route("/api/sessions/<sid>/export")
def session_export(sid):
    csv = face_db.export_session_csv(sid)
    buf = io.BytesIO(csv.encode())
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=f"attendance_{sid}.csv")

# ── Settings ──

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "confidence_threshold": face_db.CONFIDENCE_THRESHOLD,
        "cooldown_sec": cam.COOLDOWN_SEC,
        "capture_count": cam.capture_target,
        "unknown_cooldown_sec": cam.UNKNOWN_COOLDOWN_SEC
    })

@app.route("/api/settings", methods=["POST"])
def save_settings():
    data = request.json
    if "confidence_threshold" in data:
        face_db.CONFIDENCE_THRESHOLD = int(data["confidence_threshold"])
    if "cooldown_sec" in data:
        cam.COOLDOWN_SEC = int(data["cooldown_sec"])
    if "capture_count" in data:
        cam.capture_target = int(data["capture_count"])
    if "unknown_cooldown_sec" in data:
        cam.UNKNOWN_COOLDOWN_SEC = int(data["unknown_cooldown_sec"])
    return jsonify({"ok": True})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cam.reload_model()
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │   Proctor System V2  →  http://localhost:5000  │")
    print("  └─────────────────────────────────────────┘\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

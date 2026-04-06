import os
import time
import cv2
import urllib.request
from flask import Flask, Response, render_template
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from behavior import BehaviorAnalyzer
from logger import Logger

# ─── Flask Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ─── Directories & Modules ─────────────────────────────────────────────────────
os.makedirs('alerts', exist_ok=True)
logger = Logger('log.csv')
behavior = BehaviorAnalyzer(target_threshold=5, window_sec=10, cooldown_sec=5)

# ─── MediaPipe Tasks API Setup ─────────────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

if not os.path.isfile(MODEL_PATH):
    print(f'Downloading MediaPipe face landmarker model to {MODEL_PATH}...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Model downloaded OK.')

BaseOptions = mp.tasks.BaseOptions

_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = FaceLandmarker.create_from_options(_options)

# ─── Camera Setup ──────────────────────────────────────────────────────────────
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', '0')
video_source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
camera = cv2.VideoCapture(video_source)

# Force MJPEG codec — fixes green/corrupted frames from virtual cameras like DroidCam
if isinstance(video_source, int):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)



# ─── Helper Functions ──────────────────────────────────────────────────────────
def estimate_direction(nose_x: int, x1: int, x2: int) -> str:
    """Determine head direction based on nose position relative to face bounding box."""
    face_width = x2 - x1
    if face_width == 0:
        return 'CENTER'
    ratio = (nose_x - x1) / face_width
    if ratio < 0.38:
        return 'RIGHT'
    elif ratio > 0.62:
        return 'LEFT'
    return 'CENTER'


def draw_ui(frame, suspicious: bool, recent_movements: int, msg_time: float) -> None:
    """Render stats panel and alert overlays onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent stats panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 165), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    status_color = (0, 0, 255) if suspicious else (0, 255, 0)
    status_txt = "SUSPICIOUS" if suspicious else "NORMAL"

    cv2.putText(frame, f"LEFT:   {behavior.left_count}",   (20,  42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"RIGHT:  {behavior.right_count}",  (20,  72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"CENTER: {behavior.center_count}", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"STATUS: {status_txt}",            (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color,    2)
    cv2.putText(frame, f"Recent L/R: {recent_movements}/{behavior.threshold}",
                (190, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if suspicious:
        cv2.putText(frame, "!! SUSPICIOUS DETECTED !!",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    if time.time() - msg_time < 2.0:
        cv2.putText(frame, "IMAGE SAVED",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


# ─── Frame Generator ───────────────────────────────────────────────────────────
def generate_frames():
    msg_time = 0.0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h_img, w_img = frame.shape[:2]

        # Convert to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        detection_result = face_landmarker.detect(mp_image)
        face_directions = []

        if detection_result.face_landmarks:
            for face_lm_list in detection_result.face_landmarks:
                # Build bounding box from landmarks
                xs = [lm.x for lm in face_lm_list]
                ys = [lm.y for lm in face_lm_list]
                x1, y1 = int(min(xs) * w_img), int(min(ys) * h_img)
                x2, y2 = int(max(xs) * w_img), int(max(ys) * h_img)

                # Nose tip = landmark index 1
                nose_x = int(face_lm_list[1].x * w_img)
                nose_y = int(face_lm_list[1].y * h_img)

                direction = estimate_direction(nose_x, x1, x2)
                face_directions.append(direction)

                box_color = (0, 255, 0) if direction == 'CENTER' else (0, 165, 255)
                cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), box_color, 2)
                cv2.putText(frame, direction, (x1, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
                cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 255), -1)

        # Update behavior engine
        behavior.update(face_directions)
        suspicious = behavior.is_suspicious()
        recent_movements = behavior.get_recent_movements_count()

        # Alert capture with cooldown
        if suspicious and behavior.can_capture():
            behavior.trigger_capture()
            ts = int(time.time())
            filename = f"alerts/alert_{ts}.jpg"
            cv2.imwrite(filename, frame)
            logger.log_event("Excessive head movement")
            msg_time = time.time()
            print(f"[ALERT] Image saved → {filename}")

        draw_ui(frame, suspicious, recent_movements, msg_time)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

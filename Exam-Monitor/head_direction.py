import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Nose tip (approx landmark 1)
            nose = face_landmarks.landmark[1]

            h, w, _ = frame.shape
            x = int(nose.x * w)

            if x < w//3:
                direction = "LEFT"
            elif x > 2*w//3:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            cv2.putText(frame, direction, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Head Direction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def liveness_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False  # no face detected

        # Get eye landmarks (left & right eye indices from face mesh spec)
        left_eye = [33, 133]
        right_eye = [362, 263]

        for landmarks in results.multi_face_landmarks:
            lx = int(landmarks.landmark[left_eye[0]].x * w)
            ly = int(landmarks.landmark[left_eye[0]].y * h)
            rx = int(landmarks.landmark[right_eye[0]].x * w)
            ry = int(landmarks.landmark[right_eye[0]].y * h)

            # Simple distance check (blinking if distance reduces drastically)
            eye_dist = abs(ly - ry)
            if eye_dist < 5:   # threshold for blink
                return True

        return True  # if face detected, assume real

import os
import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

# ------------- CONFIG -------------
FACES_DIR = "faces"
COSINE_THRESHOLD = 0.95  # ปรับให้เข้มขึ้น
KEY_LANDMARKS = [
    33, 133, 160, 159, 158, 144,         # left eye
    362, 263, 387, 386, 385, 373,        # right eye
    1, 2, 98, 327,                       # nose bridge
    61, 291, 78, 308, 13, 14,            # mouth
    199, 234, 93, 132, 58, 288,          # jawline
    10, 152                              # forehead center, chin center
]
# -----------------------------------

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------- STEP 1: Create Embedding from Folder ----------
def create_known_faces():
    print("🔍 Creating face vectors from /faces ...")
    known_faces = {}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for person in os.listdir(FACES_DIR):
            person_path = os.path.join(FACES_DIR, person)
            if not os.path.isdir(person_path):
                continue
            known_faces[person] = []

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    vector = extract_key_vector(landmarks)
                    known_faces[person].append(vector)
                    print(f"✅ Added: {img_path}")
                else:
                    print(f"❌ No face: {img_path}")

    return known_faces

# ---------- STEP 2: Extract & Normalize Landmarks ----------
def extract_key_vector(landmarks):
    key_points = np.array([[landmarks[i].x, landmarks[i].y] for i in KEY_LANDMARKS])
    center = np.mean(key_points, axis=0)
    normed = key_points - center
    flat = normed.flatten()
    return flat / np.linalg.norm(flat)

# ---------- STEP 3: Compare by Cosine ----------
def identify_by_cosine(vec, known_faces, threshold=COSINE_THRESHOLD):
    best_score = -1
    best_name = "Unknown"

    for name, vectors in known_faces.items():
        for known_vec in vectors:
            score = cosine_similarity(vec.reshape(1, -1), known_vec.reshape(1, -1))[0][0]
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return "Unknown", best_score

# ---------- STEP 4: Webcam Loop ----------
def run_webcam_recognition(known_faces):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    prev_time = 0

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = max(0, int(box.xmin * w))
                    y1 = max(0, int(box.ymin * h))
                    x2 = min(w, int((box.xmin + box.width) * w))
                    y2 = min(h, int((box.ymin + box.height) * h))

                    # Skip face too small
                    if (x2 - x1) < 50 or (y2 - y1) < 50:
                        continue

                    face_roi = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mesh_result = face_mesh.process(face_rgb)

                    name = "Unknown"
                    score = 0

                    if mesh_result.multi_face_landmarks:
                        landmarks = mesh_result.multi_face_landmarks[0].landmark
                        vector = extract_key_vector(landmarks)
                        name, score = identify_by_cosine(vector, known_faces)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Face Recognition (Optimized)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- RUN ------------------
if __name__ == "__main__":
    known_faces = create_known_faces()
    run_webcam_recognition(known_faces)

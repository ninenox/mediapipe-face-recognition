import os
import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

# ------------- CONFIG -------------
# กำหนดพาธโฟลเดอร์ฐานของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ใช้พาธสัมบูรณ์เพื่อให้เรียกสคริปต์จากที่ใดก็ได้
FACES_DIR = os.path.join(BASE_DIR, "faces")
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

    if not os.path.exists(FACES_DIR):
        raise FileNotFoundError(f"Folder {FACES_DIR} not found")

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

# ---------- STEP 3.5: Register New Face ----------
def register_new_face(cap, known_faces, num_samples=5, delay=1):
    """Capture a new face from webcam and update known faces."""
    name = input("🆕 กรอกชื่อสำหรับใบหน้าใหม่: ").strip()
    if not name:
        print("❌ ไม่ได้กรอกชื่อ ยกเลิกการบันทึก")
        return known_faces

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    saved_files = []

    print("📸 เริ่มถ่ายภาพ...")
    for i in range(num_samples):
        time.sleep(delay)
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถถ่ายภาพได้")
            continue
        file_path = os.path.join(person_dir, f"{int(time.time())}_{i}.jpg")
        cv2.imwrite(file_path, frame)
        saved_files.append(file_path)
        print(f"✅ บันทึก: {file_path}")

    if saved_files:
        print("✨ อัปเดตฐานข้อมูล ...")
        known_faces = create_known_faces()
        print("✅ เสร็จสิ้น")
    else:
        print("❌ ไม่มีไฟล์ที่ถูกบันทึก")

    return known_faces

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
            cv2.putText(frame, "กด 'n' ลงทะเบียนใบหน้าใหม่", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Recognition (Optimized)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                known_faces = register_new_face(cap, known_faces)

    cap.release()
    cv2.destroyAllWindows()

class WebcamRecognition:
    def __init__(self, known_faces, frame_callback=None):
        self.known_faces = known_faces
        self.frame_callback = frame_callback or self.default_callback
        self.running = False
        self.cap = None

    def default_callback(self, frame):
        cv2.imshow("Face Recognition (Optimized)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.running = True
        prev_time = 0

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector, \
             mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5) as face_mesh:

            while self.running:
                ret, frame = self.cap.read()
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
                            name, score = identify_by_cosine(vector, self.known_faces)

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(frame, f'FPS: {int(fps)}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                self.frame_callback(frame)

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


# ----------------- RUN ------------------
if __name__ == "__main__":
    known_faces = create_known_faces()
    webcam = WebcamRecognition(known_faces)
    webcam.start()

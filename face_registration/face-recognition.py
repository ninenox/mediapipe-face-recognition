import os
import argparse
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
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", 0.95))  # ปรับให้เข้มขึ้น

# จุด landmark สำคัญเพิ่มเติมและน้ำหนักของแต่ละจุด
KEY_LANDMARKS = [
    33, 133, 160, 159, 158, 144,         # left eye
    362, 263, 387, 386, 385, 373,        # right eye
    107, 336, 50, 280,                   # temples
    1, 2, 98, 327,                       # nose bridge
    61, 291, 78, 308, 13, 14,            # mouth
    199, 234, 93, 132, 58, 288,          # jawline
    10, 152                              # forehead center, chin center
]
# ให้ดวงตาและปากมีอิทธิพลมากกว่า
KEY_WEIGHTS = np.array([
    *[2.0]*6,        # left eye
    *[2.0]*6,        # right eye
    *[1.5]*4,        # temples
    *[1.5]*4,        # nose bridge
    *[2.0]*6,        # mouth
    *[1.0]*6,        # jawline
    *[1.0]*2         # forehead & chin
])
# -----------------------------------

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------- STEP 1: Create Embedding from Folder ----------
def create_known_faces(use_median=False, keep_full=False):
    """โหลดเวกเตอร์ใบหน้าทั้งหมดจากโฟลเดอร์แล้วสร้างตัวแทนหนึ่งเวกเตอร์ต่อคน.

    Parameters
    ----------
    use_median : bool, optional
        หาก True จะใช้ค่า median แทนค่าเฉลี่ยในการรวมเวกเตอร์ (ค่าเริ่มต้น False)
    keep_full : bool, optional
        หาก True จะเก็บรายการเวกเตอร์ทั้งหมดไว้ในผลลัพธ์เพื่อการทดลอง

    Returns
    -------
    dict
        {ชื่อคน: เวกเตอร์ตัวแทน หรือ {"rep": เวกเตอร์เฉลี่ย, "vectors": [...]} }
    """
    print("🔍 Creating face vectors from /faces ...")
    known_faces = {}

    if not os.path.exists(FACES_DIR):
        raise FileNotFoundError(f"Folder {FACES_DIR} not found")

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for person in os.listdir(FACES_DIR):
            person_path = os.path.join(FACES_DIR, person)
            if not os.path.isdir(person_path):
                continue

            vectors = []

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
                    vectors.append(vector)
                    print(f"✅ Added: {img_path}")
                else:
                    print(f"❌ No face: {img_path}")

            if not vectors:
                continue

            rep = np.median(vectors, axis=0) if use_median else np.mean(vectors, axis=0)
            norm = np.linalg.norm(rep)
            rep = rep / norm if norm != 0 else rep

            if keep_full:
                known_faces[person] = {"rep": rep, "vectors": vectors}
            else:
                known_faces[person] = rep

    return known_faces

# ---------- STEP 2: Extract & Normalize Landmarks ----------
def extract_key_vector(landmarks):
    """ดึงและจัดแนวพิกัด landmark ให้เป็นเวกเตอร์มาตรฐาน"""
    key_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in KEY_LANDMARKS])

    # จัดศูนย์กลางโดยใช้น้ำหนัก
    center = np.average(key_points, axis=0, weights=KEY_WEIGHTS)
    normed = key_points - center

    # จัดแนวแกนหลักเพื่อลดผลกระทบจากการเอียงศีรษะ (Procrustes แบบง่าย)
    U, _, _ = np.linalg.svd(normed.T)
    aligned = normed @ U

    # ทำ normalization และถ่วงน้ำหนักก่อน flatten
    weighted = aligned * KEY_WEIGHTS[:, None]
    flat = weighted.flatten()
    norm = np.linalg.norm(flat)
    return flat / norm if norm != 0 else flat

# ---------- STEP 3: Compare by Cosine ----------
def identify_by_cosine(vec, known_faces, threshold=None, margin=0.03, use_full=False):
    """Identify the closest known face using cosine similarity.

    Parameters
    ----------
    vec : np.ndarray
        Vector representation of the face to identify.
    known_faces : dict
        Dictionary of known face representations.
    threshold : float, optional
        Minimum cosine similarity required for a match.
    margin : float, optional
        Minimum difference between the best and second-best scores.
    use_full : bool, optional
        If True, use all stored vectors for comparison when available.
    """
    threshold = threshold if threshold is not None else COSINE_THRESHOLD

    scores = []
    for name, data in known_faces.items():
        if use_full and isinstance(data, dict) and "vectors" in data:
            vectors = data["vectors"]
        else:
            rep = data["rep"] if isinstance(data, dict) else data
            vectors = [rep]
        for known_vec in vectors:
            score = cosine_similarity(vec.reshape(1, -1), known_vec.reshape(1, -1))[0][0]
            scores.append((score, name))

    if not scores:
        return "Unknown", -1

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1

    if best_score < threshold or (best_score - second_score) < margin:
        return "Unknown", best_score
    return best_name, best_score

# ---------- STEP 3.5: Register New Face ----------
def register_new_face(cap, known_faces, num_samples=5, delay=1, use_median=False, keep_full=False):
    """Capture a new face from webcam and update known faces."""
    name = input("🆕 กรอกชื่อสำหรับใบหน้าใหม่: ").strip()
    if not name:
        print("❌ ไม่ได้กรอกชื่อ ยกเลิกการบันทึก")
        return known_faces

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    saved_files = []

    print("📸 เริ่มถ่ายภาพ...")
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        for i in range(num_samples):
            time.sleep(delay)
            ret, frame = cap.read()
            if not ret:
                print("❌ ไม่สามารถถ่ายภาพได้")
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            if not result.detections:
                print("❌ ไม่พบใบหน้า ข้ามภาพนี้")
                continue

            box = result.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                print("❌ ไม่ได้ภาพใบหน้าที่ใช้ได้")
                continue

            face_resized = cv2.resize(face_roi, (256, 256))
            file_path = os.path.join(person_dir, f"{int(time.time())}_{i}.jpg")
            cv2.imwrite(file_path, face_resized)
            saved_files.append(file_path)
            print(f"✅ บันทึก: {file_path}")

    if saved_files:
        print("✨ อัปเดตฐานข้อมูล ...")
        known_faces = create_known_faces(use_median=use_median, keep_full=keep_full)
        print("✅ เสร็จสิ้น")
    else:
        print("❌ ไม่มีไฟล์ที่ถูกบันทึก")

    return known_faces

# ---------- STEP 4: Webcam Loop ----------

def run_webcam_recognition(known_faces, threshold=COSINE_THRESHOLD):
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
                        name, score = identify_by_cosine(vector, known_faces, threshold=threshold)

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
    def __init__(self, known_faces, frame_callback=None, threshold=COSINE_THRESHOLD):
        self.known_faces = known_faces
        self.frame_callback = frame_callback or self.default_callback
        self.threshold = threshold
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
                            name, score = identify_by_cosine(vector, self.known_faces, threshold=self.threshold)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                        help="Cosine similarity threshold (0-1)")
    args = parser.parse_args()
    COSINE_THRESHOLD = args.threshold
    known_faces = create_known_faces()
    webcam = WebcamRecognition(known_faces, threshold=COSINE_THRESHOLD)
    webcam.start()

import os
import csv
import time
from datetime import datetime
import cv2
import mediapipe as mp
import importlib.util

# โหลดฟังก์ชันจาก face-recognition.py แบบไดนามิก (ชื่อไฟล์มีขีดกลาง)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_recog_path = os.path.join(BASE_DIR, "face-recognition.py")
spec = importlib.util.spec_from_file_location("face_recog", face_recog_path)
face_recog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_recog)

ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
LOG_COOLDOWN = 300  # หน่วงเวลา 5 นาทีไม่ให้บันทึกซ้ำ

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def mark_attendance(name: str) -> str | None:
    """บันทึกเวลาเข้า/ออกลงไฟล์ CSV
    คืนค่า 'IN' หรือ 'OUT' หากบันทึกสำเร็จ, คืน None หากบันทึกซ้ำ"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    status = "IN"
    records = []

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)
            for row in reversed(records):
                if row["date"] == date_str and row["name"] == name:
                    if row["status"] == "IN":
                        status = "OUT"
                        break
                    if row["status"] == "OUT":
                        return None

    file_exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "time", "name", "status"])
        writer.writerow([date_str, time_str, name, status])
    return status


def run_attendance() -> None:
    known_faces = face_recog.create_known_faces()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    last_logged: dict[str, float] = {}

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector, \
            mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

                    if (x2 - x1) < 50 or (y2 - y1) < 50:
                        continue

                    face_roi = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mesh_result = face_mesh.process(face_rgb)

                    name = "Unknown"
                    status_text = ""

                    if mesh_result.multi_face_landmarks:
                        landmarks = mesh_result.multi_face_landmarks[0].landmark
                        vector = face_recog.extract_key_vector(landmarks)
                        name, _ = face_recog.identify_by_cosine(vector, known_faces)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if name != "Unknown":
                        now = time.time()
                        last_time = last_logged.get(name, 0)
                        if now - last_time > LOG_COOLDOWN:
                            status = mark_attendance(name)
                            if status:
                                status_text = status
                                last_logged[name] = now

                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if status_text:
                        cv2.putText(frame, status_text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, "กด q เพื่อออก", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()

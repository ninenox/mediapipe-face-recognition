import cv2
import mediapipe as mp
import time  # <🔸 เพิ่มสำหรับจับเวลา>

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# เริ่มต้นตัวแปรเวลา
prev_time = 0

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # จับเวลาปัจจุบัน
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # แปลง BGR เป็น RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        # วาดกรอบใบหน้า
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # วาด FPS ที่มุมซ้ายบน
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

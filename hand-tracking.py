import cv2
import mediapipe as mp
import time

# เรียกใช้งานโมดูล MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# กล้อง
cap = cv2.VideoCapture(0)

# จับเวลา FPS
prev_time = 0

# สร้างตัวตรวจจับมือ
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # คำนวณ FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # แปลง BGR เป็น RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ตรวจจับมือ
        results = hands.process(img_rgb)

        # วาดผลลัพธ์ (keypoints + เส้นเชื่อม)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # แสดง FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # แสดงภาพ
        cv2.imshow("MediaPipe Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()


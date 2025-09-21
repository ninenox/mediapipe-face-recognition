# mediapipe-example

<p align="center">
  <img src="https://github.com/google/mediapipe/blob/master/docs/images/mediapipe_logo_color.png?raw=1" alt="MediaPipe Logo" width="200">
</p>

This project collects sample scripts using **MediaPipe** and `OpenCV` for real-time image analysis such as face detection, Face Mesh, hand tracking, pose detection, and face recognition. They can run on a Raspberry Pi 5 at about 15–25 FPS.

## Installation

You should install Python 3 and then install the libraries listed in `requirements.txt` with the command:

```bash
pip install -r requirements.txt
```

## Folder structure

- `face-detect.py` – face detection script
- `face-mesh.py` – draw Face Mesh points with FPS counter
- `hand-tracking.py` – detect and track hands
- `pose-detect.py` – detect body poses
- `face_registration/` – face recognition and attendance system
  - `face-recognition.py` – face recognition using features from Face Mesh
  - `face_recognition_ui.py` – GUI for face recognition
  - `attendance.py` – face recognition for logging employee check‑in/out times
  - `faces/` – sample images for building face vectors used for recognition
    Inside this folder create a subfolder for each person, e.g. `faces/person-name/`
    Add several face images of that person (supports `.jpg` or `.png`) before running `face-recognition.py`
    If `faces/` does not exist, create it first with `mkdir faces`

## Usage

### Quick launcher

Run `run_demo.py` to list available demos and execute them without remembering file paths. Extra arguments after `--` are forwarded to the target script.

```bash
python run_demo.py list
python run_demo.py run face-detect
python run_demo.py run face-registration -- --threshold 0.92
```

### Direct execution

You can still run the scripts manually if you prefer:

```bash
python face-detect.py
python face-mesh.py
python hand-tracking.py
python pose-detect.py
python face_registration/face-recognition.py
python face_registration/face_recognition_ui.py
python face_registration/attendance.py
```

Before running `face_registration/face_recognition_ui.py`, install `Pillow` (for example, with `pip install Pillow`) and verify that `face_registration/face_recognition_processor.py` is present in the project.

While running `face_registration/face-recognition.py`, press `n` to capture and register a new face. The image will be saved and the `face_registration/faces/` folder will be updated automatically.

For `face_registration/face-recognition.py` you can adjust the cosine similarity threshold used for matching by either setting the environment variable `COSINE_THRESHOLD` or passing the command-line option `--threshold`, for example:

```bash
COSINE_THRESHOLD=0.9 python face_registration/face-recognition.py
# or
python face_registration/face-recognition.py --threshold 0.9
```

Press `q` to close each program's display window.

## Notes

These scripts are intended for experimentation or studying MediaPipe. Users can adjust parameters within each script to suit their own tasks.

## ไอเดียต่อยอดเพื่อให้โปรเจ็กต์น่าสนใจยิ่งขึ้น

- เพิ่มหน้าจอ Dashboard เล็ก ๆ (Streamlit หรือ Gradio) เพื่อโชว์ผลลัพธ์ของแต่ละเดโมพร้อมวิดีโอตัวอย่าง
- เขียน Dockerfile สำหรับรันบนเครื่องที่ไม่มี Python environment พร้อมใช้งาน
- สร้างโมดูลฝึกโมเดลเฉพาะ (เช่น ท่าทางมือสำหรับสั่งงาน) แล้วนำมาเชื่อมกับ `run_demo.py`
- บันทึกค่า FPS, อุณหภูมิของ Raspberry Pi และบันทึกเป็นกราฟเพื่อเทียบประสิทธิภาพ
- ทำ workflow สำหรับ CI เช่น GitHub Actions เพื่อตรวจสอบว่าทุกสคริปต์ import ได้โดยไม่ error

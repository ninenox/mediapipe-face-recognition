# mediapipe-example

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
- `face-recognition.py` – face recognition using features from Face Mesh
- `faces/` – sample images for building face vectors used for recognition
  Inside this folder create a subfolder for each person, e.g. `faces/person-name/`
  Add several face images of that person (supports `.jpg` or `.png`) before running `face-recognition.py`
  If `faces/` does not exist, create it first with `mkdir faces`

## Usage

Run the desired script using commands like:

```bash
python face-detect.py
python face-mesh.py
python hand-tracking.py
python pose-detect.py
python face-recognition.py
```

Press `q` to close each program's display window.

## Notes

These scripts are intended for experimentation or studying MediaPipe. Users can adjust parameters within each script to suit their own tasks.

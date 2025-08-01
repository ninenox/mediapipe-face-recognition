import threading
from typing import Callable, Optional

import cv2


class FaceRecognitionProcessor:
    """Simple webcam loop that streams frames via a callback."""

    def __init__(self, frame_callback: Callable[[any], None]) -> None:
        self.frame_callback = frame_callback
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start capturing frames from the webcam."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_callback(frame)

    def stop(self) -> None:
        """Stop capturing frames."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None

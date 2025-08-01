import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2

from face_recognition_processor import FaceRecognitionProcessor


class FaceRecognitionUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Face Recognition UI")
        self.processor: FaceRecognitionProcessor | None = None

        self.image_label = tk.Label(master)
        self.image_label.pack()

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=10)

        self.start_button = tk.Button(btn_frame, text="เริ่มกล้อง", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(btn_frame, text="หยุด", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def start_camera(self) -> None:
        if self.processor:
            return
        try:
            self.processor = FaceRecognitionProcessor(self.update_image)
            self.processor.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def stop_camera(self) -> None:
        if self.processor:
            self.processor.stop()
            self.processor = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_image(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)


def main() -> None:
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

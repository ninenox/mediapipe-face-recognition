import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2



class FaceRecognitionUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Face Recognition UI")
        self.cap = None
        self.running = False

        self.image_label = tk.Label(master)
        self.image_label.pack()

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=10)

        self.start_button = tk.Button(btn_frame, text="เริ่มกล้อง", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(btn_frame, text="หยุด", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def start_camera(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror("Error", "Cannot open camera")
            return
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.master.after(10, self._update_frame)

    def stop_camera(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def _update_frame(self) -> None:
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self.update_image(frame)
        if self.running:
            self.master.after(10, self._update_frame)

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

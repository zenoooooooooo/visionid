import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Camera failed to open. Check GStreamer pipeline.")

    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
    ):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Frame not received")
            return None
        return frame

    def release(self):
        self.cap.release()

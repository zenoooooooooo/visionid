from utils.camera import Camera
from utils.face_detector import FaceDetector
import cv2

cam = Camera()
detector = FaceDetector()

while True:
    frame = cam.read()
    if frame is None:
        break

    faces = detector.detect(frame)

    for (x1, y1, x2, y2, confidence) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("VisionID", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

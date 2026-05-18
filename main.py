from utils.camera import Camera
from utils.face_detector import FaceDetector
from utils.face_recognizer import FaceRecognizer
from utils.pipeline import Pipeline
from stream_server import stream_server, start_server
from datetime import datetime

import cv2

def enroll_mode(cam, detector, recognizer):
    name = input("Enter name to enroll: ").strip()
    if not name:
        print("Name cannot be empty!")
        return

    target = int(input("How many captures? (recommended 10): ").strip() or "10")
    print(f"Enrolling {name}... Press SPACE to capture ({target} needed), Q to quit")

    captured = 0
    while captured < target:
        frame = cam.read()
        if frame is None:
            break

        faces = detector.detect(frame)

        for (x1, y1, x2, y2, confidence) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {captured}/{target}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"SPACE=capture Q=quit [{captured}/{target}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("VisionID - Enroll", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if faces:
                x1, y1, x2, y2, _ = faces[0]
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    print("Face too small, try again")
                    continue
                recognizer.enroll(name, face_img)
                captured += 1
                print(f"Captured {captured}/{target}")
            else:
                print("No face detected, try again")

    if captured == target:
        print(f"Enrollment complete for {name} with {captured} captures!")
    else:
        print(f"Enrollment incomplete: {captured}/{target} captures saved")

    cv2.destroyAllWindows()

def recognize_mode(cam, detector, recognizer):
    print("Recognition mode. Press Q to quit")
    pipeline = Pipeline(cam, detector, recognizer)

    while True:
        frame, results = pipeline.get()
        if frame is None:
            continue

        for (x1, y1, x2, y2, name, score) in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stream_server.update_recognition(name, score, timestamp)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h, w = frame.shape[:2]
        cv2.putText(frame, timestamp, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        stream_server.update_frame(frame)
        cv2.imshow("VisionID - Recognize", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

def main():
    print("=== VisionID ===")
    print("1. Enroll face")
    print("2. Recognize faces")
    choice = input("Select mode (1 or 2): ").strip()

    start_server()

    cam = Camera()
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    try:
        if choice == "1":
            enroll_mode(cam, detector, recognizer)
        elif choice == "2":
            recognize_mode(cam, detector, recognizer)
        else:
            print("Invalid choice!")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

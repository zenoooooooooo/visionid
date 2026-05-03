import cv2

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNet(
            "models/face_detection/deploy.prototxt",
            "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")
                faces.append((x1, y1, x2, y2, float(confidence)))

        return faces

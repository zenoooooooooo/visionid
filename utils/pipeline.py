import cv2
import threading
import queue
import numpy as np

class Pipeline:
    def __init__(self, cam, detector, recognizer):
        self.cam = cam
        self.detector = detector
        self.recognizer = recognizer

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.latest_frame = None
        self.latest_results = []
        self.lock = threading.Lock()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

        self.capture_thread.start()
        self.inference_thread.start()

    def _capture_loop(self):
        while self.running:
            frame = self.cam.read()
            if frame is None:
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            with self.lock:
                self.latest_frame = frame

    def _nms(self, faces, overlap_threshold=0.3):
        if not faces:
            return []

        boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in faces])
        scores = np.array([conf for (_, _, _, _, conf) in faces])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[order[1:]]

            order = order[np.where(overlap <= overlap_threshold)[0] + 1]

        return [faces[i] for i in keep]

    def _inference_loop(self):
        frame_count = 0
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            faces = self.detector.detect(frame)
            faces = self._nms(faces)  
            results = []

            for (x1, y1, x2, y2, confidence) in faces:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                name, score = self.recognizer.recognize(face_img)
                results.append((x1, y1, x2, y2, name, score))

            with self.lock:
                self.latest_results = results

    def get(self):
        with self.lock:
            frame = self.latest_frame
            results = list(self.latest_results)
        return frame, results

    def stop(self):
        self.running = False

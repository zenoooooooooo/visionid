import cv2
import numpy as np

class TrackedFace:
    def __init__(self, track_id, name, score, bbox):
        self.track_id = track_id
        self.name = name
        self.score = score
        self.bbox = bbox
        self.lost = False
        self.frames_since_recognition = 0

class FaceTracker:
    def __init__(self, detector, recognizer, detect_every=5, rerecognize_every=30):
        self.detector = detector
        self.recognizer = recognizer
        self.tracked_faces = []
        self.next_id = 1
        self.detect_every = detect_every
        self.rerecognize_every = rerecognize_every
        self.frame_count = 0

    def _iou(self, boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def update(self, frame):
        self.frame_count += 1

        # Only run detection every N frames
        if self.frame_count % self.detect_every == 0:
            detections = self.detector.detect(frame)

            for (x1, y1, x2, y2, confidence) in detections:
                det_box = (x1, y1, x2, y2)

                matched = None
                best_iou = 0.3
                for tf in self.tracked_faces:
                    iou = self._iou(det_box, tf.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        matched = tf

                if matched:
                    matched.bbox = det_box
                    matched.lost = False
                    matched.frames_since_recognition += 1

                    # Re-recognize every N frames
                    if matched.frames_since_recognition >= self.rerecognize_every:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            name, score = self.recognizer.recognize(face_img)
                            if name != "Unknown":
                                matched.name = name
                                matched.score = score
                        matched.frames_since_recognition = 0
                else:
                    # New face
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    name, score = self.recognizer.recognize(face_img)
                    new_face = TrackedFace(
                        track_id=self.next_id,
                        name=name,
                        score=score,
                        bbox=det_box
                    )
                    self.tracked_faces.append(new_face)
                    self.next_id += 1

            # Mark faces not seen in this detection as lost
            detected_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in detections]
            for tf in self.tracked_faces:
                matched = any(self._iou(tf.bbox, b) > 0.3 for b in detected_boxes)
                if not matched:
                    tf.frames_since_recognition += 1
                    if tf.frames_since_recognition > 30:
                        tf.lost = True

            self.tracked_faces = [tf for tf in self.tracked_faces if not tf.lost]

        return self.tracked_faces

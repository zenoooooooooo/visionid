import cv2
import numpy as np
import os
import pickle

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.reverse_label_map = {}
        self.face_data = []
        self.label_data = []

        base = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base, "../data/embeddings.pkl")
        self.model_path = os.path.join(base, "../data/lbph_model.yml")

        if os.path.exists(self.db_path) and os.path.exists(self.model_path):
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                self.label_map = data["label_map"]
                self.reverse_label_map = data["reverse_label_map"]
                self.face_data = data["face_data"]
                self.label_data = data["label_data"]
            self.recognizer.read(self.model_path)
            print(f"Loaded {len(self.label_map)} people")
        else:
            print("No model found, starting fresh")

    def _preprocess(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        equalized = cv2.equalizeHist(resized)
        return equalized

    def enroll(self, name, face_img):
        processed = self._preprocess(face_img)

        if name not in self.label_map:
            label = len(self.label_map)
            self.label_map[name] = label
            self.reverse_label_map[label] = name

        label = self.label_map[name]
        self.face_data.append(processed)
        self.label_data.append(label)

        self.recognizer.train(self.face_data, np.array(self.label_data))
        self._save()
        print(f"Enrolled: {name} (total: {self.label_data.count(label)} captures)")
        return True

    def recognize(self, face_img, threshold=110):
        if not self.label_map:
            return "Unknown", 0.0

        processed = self._preprocess(face_img)
        label, confidence = self.recognizer.predict(processed)

        print(f"Label: {label}, Confidence: {confidence:.2f}")

        if confidence < threshold:
            name = self.reverse_label_map.get(label, "Unknown")
            score = 1 - (confidence / threshold)
            return name, score
        return "Unknown", 0.0

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "label_map": self.label_map,
                "reverse_label_map": self.reverse_label_map,
                "face_data": self.face_data,
                "label_data": self.label_data
            }, f)
        self.recognizer.write(self.model_path)
        print("Model saved")

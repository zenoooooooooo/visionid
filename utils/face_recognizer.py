import cv2
import numpy as np
import os
import pickle

class FaceRecognizer:
    def __init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base, "../models/face_recognition/nn4.small2.v1.t7")

        self.net = cv2.dnn.readNetFromTorch(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.eye_cascade = cv2.CascadeClassifier(
            "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml"
        )

        self.known_embeddings = []
        self.known_names = []

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, "../data/embeddings.pkl")

        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                self.known_embeddings = data["embeddings"]
                self.known_names = data["names"]
            print(f"Loaded {len(self.known_names)} embeddings for {len(set(self.known_names))} people")
        else:
            print("No embeddings found, starting fresh")

    def _align_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]


            left_eye = (ex1 + ew1 // 2, ey1 + eh1 // 2)
            right_eye = (ex2 + ew2 // 2, ey2 + eh2 // 2)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))

            h, w = face_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(face_img, M, (w, h))
            return aligned

        return face_img

    def get_embedding(self, face_img):
        if face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return None

        aligned = self._align_face(face_img)
        blob = cv2.dnn.blobFromImage(aligned, 1.0/255, (96, 96),
                                      (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward().flatten()

    def enroll(self, name, face_img):
        embedding = self.get_embedding(face_img)
        if embedding is None:
            print("Face too small, skipping")
            return False
        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        self._save()
        print(f"Enrolled: {name} (total: {self.known_names.count(name)} captures)")
        return True

    def recognize(self, face_img, threshold=1.1):
        if not self.known_embeddings:
            return "Unknown", 0.0

        embedding = self.get_embedding(face_img)
        if embedding is None:
            return "Unknown", 0.0

        name_distances = {}
        for i, known_embedding in enumerate(self.known_embeddings):
            name = self.known_names[i]
            dist = np.linalg.norm(embedding - known_embedding)
            if name not in name_distances:
                name_distances[name] = []
            name_distances[name].append(dist)

        avg_distances = {name: np.mean(dists) for name, dists in name_distances.items()}
        print("Distances:", {name: f"{dist:.3f}" for name, dist in avg_distances.items()})

        best_name = min(avg_distances, key=avg_distances.get)
        best_dist = avg_distances[best_name]

        if best_dist < threshold:
            return best_name, 1 - best_dist
        return "Unknown", 0.0

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "embeddings": self.known_embeddings,
                "names": self.known_names
            }, f)
        print("Embeddings saved")

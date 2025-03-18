import cv2
import mediapipe as mp
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# Load FaceNet model (Pretrained on VGGFace2)
model = InceptionResnetV1(pretrained="vggface2").eval()

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

def extract_face_embeddings(image):
    """
    Detects faces and extracts a 512-dimension face embedding.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            face = image[y:y+h, x:x+w]  # Crop face
            face = cv2.resize(face, (160, 160))  # Resize for FaceNet
            face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()

            embedding = model(face).detach().numpy()
            return embedding, (x, y, w, h)

    return None, None

def compare_faces(embedding1, embedding2, threshold=0.7):
    """
    Computes cosine similarity between two face embeddings.
    Returns True if similarity is above the threshold.
    """
    similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold

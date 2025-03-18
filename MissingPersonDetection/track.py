import cv2
import pickle
import os
import numpy as np
from recognizer import extract_face_embeddings, compare_faces

# Load stored embeddings
db_path = "database/missing_persons_db.pkl"

if not os.path.exists(db_path):
    print("⚠️ No database found! Run store_embeddings.py first.")
    exit()

with open(db_path, "rb") as f:
    missing_persons_db = pickle.load(f)

# Open webcam
cap = cv2.VideoCapture(0)

# Set confidence threshold
CONFIDENCE_THRESHOLD = 50  # Only recognize if confidence > 50%

def draw_sci_fi_box(frame, name, x, y, w, h, confidence):
    """Draws a sci-fi style box with a pointer to the detected face."""
    box_color = (0, 255, 0)  # Green sci-fi box
    text_color = (255, 255, 255)  # White text

    # Convert confidence to a float to avoid TypeError
    confidence = float(confidence)

    # Draw name box above the face
    cv2.rectangle(frame, (x, y - 40), (x + w, y), box_color, -1)  
    cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Pointer from box to the face
    cv2.line(frame, (x + w // 2, y), (x + w // 2, y + h // 2), box_color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    embedding, bbox = extract_face_embeddings(frame)

    if embedding is not None:
        best_match = "Unrecognized"
        best_confidence = 0.0  

        for name, stored_embedding in missing_persons_db.items():
            similarity = np.dot(embedding, stored_embedding.T) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
            confidence = (similarity + 1) / 2 * 100  

            if confidence > best_confidence:
                best_confidence = float(confidence)  # Convert to float
                best_match = name if confidence > CONFIDENCE_THRESHOLD else "Unrecognized"

        # Show recognition box if a face is detected
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw sci-fi box
        draw_sci_fi_box(frame, best_match, x, y, w, h, best_confidence)

    cv2.imshow("Real-Time Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

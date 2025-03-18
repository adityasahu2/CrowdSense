import cv2
import pickle
import os
from recognizer import extract_face_embeddings

# Path to store embeddings
db_path = "database/missing_persons_db.pkl"
image_dir = "images/"  # Folder where missing persons' images are stored

# Dictionary to store embeddings
missing_persons_db = {}



for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]  # Extract name from filename
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        embedding, _ = extract_face_embeddings(image)

        if embedding is not None:
            missing_persons_db[name] = embedding

# Save embeddings to a file
with open(db_path, "wb") as f:
    pickle.dump(missing_persons_db, f)

print("✅ Face embeddings stored successfully!")

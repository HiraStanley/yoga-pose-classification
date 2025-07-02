import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from qdrant_client import QdrantClient, models
import os
import shutil

# Load the pretrained ResNet50 model
base_model = ResNet50(weights='imagenet')
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(img_path):
    """
    Read image from disk, preprocess, and extract a 2048-dim feature vector.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    feats = feature_model.predict(x)
    return feats.flatten()

def main():
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "yoga_poses"
    dataset_dir = r"C:\Users\hiras\OneDrive - The University of Chicago\UChicago\Computer Vision\Assignment3\yoga_image_search\train"

    class_names = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir))
    ]

    # Recreate the collection in Qdrant
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
    )

    # Process and upload each image
    point_id = 0
    for label in class_names:
        class_dir = os.path.join(dataset_dir, label)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(".png"):
                continue

            img_path = os.path.join(class_dir, fname)
            feature_vector = extract_features(img_path)
        
            # Insert feature vector and metadata into Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        payload={"label": label, "file_name": fname},
                        vector=feature_vector,
                    ),
                ],
            )
            print(f"Uploaded point {point_id}: {label}/{fname}")
            point_id += 1

if __name__ == "__main__":
    main()

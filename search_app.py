import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from qdrant_client import QdrantClient, models
import numpy as np
from PIL import Image
import cv2
import requests
import os

# File path
dataset_dir = r"C:\Users\hiras\OneDrive - The University of Chicago\UChicago\Computer Vision\Assignment3\yoga_image_search\train"

# Load pretrained ResNet50 and drop the classification head
base_model = ResNet50(weights="imagenet")
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Function to preprocess image and extract features
def extract_features(img):

    # Convert to RGB if image is grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Ensure image has 3 channels (RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    feats = feature_model.predict(x)
    return feats.flatten()

# Function to search for similar images using feature vectors
def search_by_vector(feature_vector):
    collection_name = "yoga_poses"
    qdrant_client = QdrantClient("http://localhost:6333")
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=feature_vector,
        query_filter=None, 
        limit=10  
    )
    return search_result

# Streamlit app
def main():
    st.title('Yoga Poses - Similar Image Search')

    # Image upload and feature extraction
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image')
        image = Image.open(uploaded_file)
        # Convert the PIL Image to a NumPy array
        image = np.array(image)

        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Extract feature vector from image
        feature_vector = extract_features(image)

        # Search using feature vector
        search_results = search_by_vector(feature_vector)
        
        if search_results:
            st.write("Search Results:")
            # Collect all image paths and labels
            images = []
            captions = []
            for result in search_results:
                label = result.payload["label"]
                fname = result.payload["file_name"]

                # Path to image in dataset directory  
                image_path = os.path.join(dataset_dir, label, fname)
                images.append(image_path)
                captions.append(label)
            
            # Display images in a grid
            st.image(images, caption=captions, width=120)

        else:
            st.write("No similar images found.")

if __name__ == '__main__':
    main()

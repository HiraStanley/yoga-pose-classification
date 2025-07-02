# Image-Based Search App

This is a simple image-based search app built with Streamlit and Qdrant. The app allows users to search for similar images in the yoga dataset using a provided query image.

### Features

- Upload an image as a query for similarity search.
- Display similar images from the yoga dataset based on the query image.
- View detailed information about the selected image.

### Installation

docker pull qdrant/qdrant #download the image
docker run -p 6333:6333 qdrant/qdrant #start the container

cd image-search
pip install -r requirements.txt

### Command
python loader.py
streamlit run image-search.py

# 🧘‍♀️ Yoga Pose Classification

This repository documents a multi-part deep learning project focused on classifying yoga poses from images. The project progresses from building a custom CNN to applying transfer learning, model quantization for real-time deployment, neural style transfer for creative exploration, and interactive deployment with Streamlit.

---

## 🚀 Project Overview

The goal of this project is to build an end-to-end machine learning pipeline that accurately classifies yoga poses, explores creative image transformations, and provides real-time interactive prediction through web deployment.

**Project Phases:**
- **Part 1:** Build a CNN from scratch and deploy using Streamlit.
- **Part 2:** Apply transfer learning with EfficientNet, perform model quantization, and deploy using Streamlit.
- **Part 3:** Experiment with neural style transfer.

---

## 📂 File Structure

- `EDA_and_CNN_from_scratch.ipynb` — Custom-built CNN trained from scratch with Streamlit deployment.
- `EfficientNet_and_YOLO.ipynb` — Transfer learning with EfficientNet, model quantization, and Streamlit deployment.
- `StyleTransfer.ipynb` — Image style transfer for yoga pose images.
- `image_search` — Image-based search app built with Streamlit and Qdrant. The app allows users to search for similar images in the yoga dataset using a provided query image.
- `tflite` — Quantized TFlite files for the two models.
- `yoga_webapp_hirastanley.py` — Run Streamlit app.
- `yoga-poses-english.txt` — Translation reference for traditional yoga pose names to English.
- `Screenshots_HiraStanley_WebApp.pdf` — Screenshots of the Streamlit app if you don't want to run it yourself!

---

## 🛠️ Part 1: CNN from Scratch

- Built a custom convolutional neural network (CNN) using base Keras layers.
- Tuned architecture, learning rate, dropout, and batch size to optimize performance.
- Deployed the trained model on **Streamlit** for real-time, web-based predictions.

**Key Learnings:**
- Gained foundational experience in building CNNs without pre-trained models.
- Learned the full cycle from model training to interactive deployment.
- Developed early insights into model limitations and overfitting risks.

---

## 🏗️ Part 2: Transfer Learning & Model Quantization

- Applied transfer learning using **EfficientNetB0** as a pre-trained base model.
- Fine-tuned the upper layers to adapt to the yoga pose dataset.
- Performed **model quantization** using TensorFlow Lite to optimize the model for mobile and low-latency applications.
- Deployed the transfer learning model on **Streamlit** to enable fast, real-time pose predictions through a user-friendly web interface.

**Key Outcomes:**
- Significantly improved model accuracy compared to the from-scratch CNN.
- Achieved model size reduction through quantization while maintaining accuracy.
- Enabled real-time, interactive predictions on a lightweight web platform.

---

## 🎨 Part 3: Neural Style Transfer

- Applied **neural style transfer** to overlay artistic styles onto yoga pose images.
- Used a pre-trained VGG network to separate and recombine content and style representations.
- Generated visually creative outputs while preserving the pose structure.

**Key Outcomes:**
- Explored the balance between content and style layers to fine-tune image aesthetics.
- Extended the project into creative computer vision applications beyond classification.

---

## 📚 Dependencies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

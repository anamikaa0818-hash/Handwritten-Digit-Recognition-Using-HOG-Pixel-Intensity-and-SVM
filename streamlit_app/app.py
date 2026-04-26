"""
Simple Streamlit App for Handwritten Digit Recognition
Authors: Anamika, Ardra, Anugopal
"""

import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image

from feature_extraction import FeatureExtractor

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and select a model for prediction.")

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    models = {}

    try:
        with open("models/SVM.pkl", "rb") as f:
            models["SVM"] = pickle.load(f)
    except:
        pass

    try:
        with open("models/KNN.pkl", "rb") as f:
            models["KNN"] = pickle.load(f)
    except:
        pass

    try:
        with open("models/LogisticRegression.pkl", "rb") as f:
            models["Logistic Regression"] = pickle.load(f)
    except:
        pass

    try:
        with open("models/RandomForest.pkl", "rb") as f:
            models["Random Forest"] = pickle.load(f)
    except:
        pass

    return models


models = load_models()
with open("models/pca_fitted.pkl", "rb") as f:
    models["PCA"] = pickle.load(f)

# =========================================================
# MODEL SELECTION
# =========================================================

if len(models) == 0:
    st.error("❌ No trained models found inside models/ folder")
    st.stop()

selected_model = st.selectbox(
    "Select Model",
    list(models.keys())
)

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    # Convert to grayscale
    img = image.convert("L")

    # Convert to numpy array
    img = np.array(img)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors if background is white
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize
    img = img / 255.0

    # Flatten
    img_flat = img.flatten()

    return img, img_flat


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(img_flat):
    
    extractor = FeatureExtractor()

    # Reshape for extractor
    X = np.array([img_flat])

    # ONLY HOG FEATURES
    hog_features = extractor.extract_hog_features(X)

    return hog_features


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    img_processed, img_flat = preprocess_image(image)

    st.write("## Processed Image")
    st.image(img_processed, width=150)

    try:

        features = extract_features(img_flat)

        # Apply PCA
        pca = models["PCA"]
        features = pca.transform(features)

        model = models[selected_model]

        prediction = model.predict(features)[0]

        st.write("## Predicted Digit")
        st.success(f"{prediction}")

    except Exception as e:

        st.error(f"Error: {e}")
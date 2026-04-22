import streamlit as st
import cv2
import numpy as np
import mahotas
import joblib
from PIL import Image
import os

# --- 1. FEATURE EXTRACTION FUNCTIONS ---
# These MUST match the logic used in your training notebook
def fd_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = 'plant_disease_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found. Please upload it to your GitHub repo.")
        return None
    return joblib.load(model_path)

# Initialize App
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection System")
st.markdown("""
Identify crop diseases instantly using **Classical Machine Learning**. 
This system uses SVM with Global Feature Extraction (Color, Texture, and Shape).
""")

data = load_model()

# --- 3. UI & PREDICTION ---
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and data is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button('🔍 Run Diagnosis'):
        with st.spinner('Extracting digital signature...'):
            try:
                # Convert PIL to OpenCV (RGB -> BGR)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Pre-processing
                img_resized = cv2.resize(opencv_image, (256, 256))
                
                # Extract Features
                fv_hu = fd_hu_moments(img_resized)
                fv_haralick = fd_haralick(img_resized)
                fv_hist = fd_histogram(img_resized)
                
                # Combine into one vector
                final_feature = np.hstack([fv_hist, fv_haralick, fv_hu]).reshape(1, -1)
                
                # Prediction Pipeline (Scaling -> PCA -> SVM)
                scaled_feat = data['scaler'].transform(final_feature)
                pca_feat = data['pca'].transform(scaled_feat)
                prediction = data['svm_model'].predict(pca_feat)
                
                # Decode Label
                result = data['label_encoder'].inverse_transform(prediction)[0]
                
                # Format result for display (e.g., Apple___Black_rot -> Apple Black rot)
                clean_result = result.replace('___', ' ').replace('_', ' ')
                
                st.success(f"### **Diagnosis:** {clean_result}")
                st.balloons()
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload an image to begin.")
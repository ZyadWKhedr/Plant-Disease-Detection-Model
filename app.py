import streamlit as st
import cv2
import numpy as np
import mahotas
import joblib
from PIL import Image
import os

# --- 1. FEATURE EXTRACTION FUNCTIONS ---
def fd_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)

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
        return None
    return joblib.load(model_path)

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="PlantDoc AI", page_icon="🌿", layout="wide")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Plant Disease Diagnosis Dashboard")
st.write("Upload a high-resolution image of a plant leaf for an instant SVM-based analysis.")

data = load_model()

if data is None:
    st.error("🚨 **Model File Missing:** Ensure 'plant_disease_model.pkl' is in the root directory.")
else:
    # --- 4. DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📸 Image Upload")
        uploaded_file = st.file_uploader("Select a JPG/PNG leaf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Specimen', use_container_width=True)
        else:
            st.info("Awaiting image upload...")

    with col2:
        st.subheader("🧪 Analysis & Results")
        if uploaded_file is not None:
            if st.button('🔍 Run Diagnostic Test'):
                with st.status("Analyzing features...", expanded=True) as status:
                    try:
                        # Processing
                        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        img_resized = cv2.resize(opencv_image, (256, 256))
                        
                        st.write("Computing HSV Histograms...")
                        fv_hist = fd_histogram(img_resized)
                        
                        st.write("Calculating Haralick Texture...")
                        fv_haralick = fd_haralick(img_resized)
                        
                        st.write("Evaluating Hu Shape Moments...")
                        fv_hu = fd_hu_moments(img_resized)
                        
                        # Pipeline
                        final_feature = np.hstack([fv_hist, fv_haralick, fv_hu]).reshape(1, -1)
                        scaled_feat = data['scaler'].transform(final_feature)
                        pca_feat = data['pca'].transform(scaled_feat)
                        prediction = data['svm_model'].predict(pca_feat)
                        
                        result = data['label_encoder'].inverse_transform(prediction)[0]
                        clean_result = result.replace('___', ' ').replace('_', ' ')
                        
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                        
                        # Result Display
                        st.markdown(f"""
                            <div class="prediction-card">
                                <p style="color: #666; margin-bottom: 5px;">DETECTION RESULT</p>
                                <h2 style="color: #2e7d32; margin-top: 0;">{clean_result}</h2>
                                <p style="font-size: 0.9em; color: #888;">Diagnostic Engine: SVM-RBF Kernel</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Diagnostic failure: {e}")
        else:
            st.warning("Please upload an image in the left panel to begin.")

# Footer
st.markdown("---")
st.caption("Developed with Python, OpenCV, and Scikit-Learn")
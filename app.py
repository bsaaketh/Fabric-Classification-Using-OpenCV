import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from PIL import Image

# LBP parameters — must match model.py
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS   # 24
LBP_METHOD = 'uniform'
LBP_BINS = LBP_POINTS + 2      # 26

model = joblib.load("fabric_classifier_model.pkl")
le = joblib.load("label_encoder.pkl")


def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.reshape(1, -1)


st.set_page_config(page_title="Fabric Classifier", page_icon="🧵", layout="centered")
st.title("🧵 Fabric Type Classifier")
st.write("Upload a fabric image to identify its material type.")

uploaded_file = st.file_uploader("Choose a fabric image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    features = extract_lbp_features(image_cv)

    pred_encoded = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    confidence = model.predict_proba(features).max() * 100

    st.success(f"**Predicted Fabric:** {pred_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show probability breakdown for all classes
    st.subheader("Class Probabilities")
    probs = model.predict_proba(features)[0]
    for cls, prob in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{cls}: {prob * 100:.1f}%")

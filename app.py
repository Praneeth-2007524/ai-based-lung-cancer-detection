from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

MODEL_PATH = Path("models/lung_cancer_model.keras")
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Lung CT Image Classification", page_icon="🫁", layout="centered")
st.title("AI-Based Lung Cancer Detection")
st.caption("Medical Image Processing + CNN | Academic prototype")
st.warning("Research/education only. This tool is not validated for diagnosis and must not guide treatment.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload a CT image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
model = load_model()

if model is None:
    st.info("The interface is ready, but no trained model was found. Arrange the dataset as shown in README.md and run `python train.py` first.")
    st.code("python train.py\nstreamlit run app.py", language="bash")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.subheader("Preprocessing preview")
    processed = ImageOps.grayscale(image)
    processed = ImageEnhance.Contrast(processed).enhance(1.4)
    processed = processed.resize(IMG_SIZE)
    st.image(processed, caption="Grayscale → contrast enhancement → 224×224 resize", width=260)

    if model is not None:
        batch = np.asarray(processed, dtype=np.float32)[None, ..., None] / 255.0
        try:
            probs = model.predict(batch, verbose=0)[0]
            idx = int(np.argmax(probs))
            st.subheader("Model output")
            st.write(f"**Predicted dataset class:** {CLASS_NAMES[idx]}")
            st.write(f"**Model score:** {float(probs[idx]) * 100:.1f}%")
            st.caption("The score is not a clinical probability and does not establish whether cancer is present.")
            st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})
        except Exception as exc:
            st.error(f"Could not run inference: {exc}")

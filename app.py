import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Model download (if not available)
file_id = "1KCzBv0387IKn1i7Jw7FB2dBWtXHefI2Z"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model function
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of the predicted class

# Sidebar navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.radio("Select Page:", ["Home", "Disease Recognition"])

# Display banner image
img = Image.open("Diseases.png")
st.image(img, use_column_width=True)

# Homepage content
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)

# Disease Recognition page
elif app_mode == "Disease Recognition":
    st.header("🔬 Plant Disease Recognition System")

    test_image = st.file_uploader("📸 Choose an Image:")

    if test_image:
        st.image(test_image, use_column_width=True)

    if st.button("🔍 Predict"):
        if test_image is not None:
            st.snow()
            st.write("✅ **Our Prediction**")
            result_index = model_prediction(test_image)

            # Class labels
            class_name = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
            st.success(f"🌿 **Model predicts:** {class_name[result_index]}")
        else:
            st.warning("⚠️ Please upload an image before predicting.")

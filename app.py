import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive model download setup
file_id = "1KCzBv0387IKn1i7Jw7FB2dBWtXHefI2Z"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Download model if not present
if not os.path.exists(model_path):
    st.warning("⏳ Downloading model from Google Drive... Please wait.")
    gdown.download(url, model_path, quiet=False)

# Function for model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set page title and layout
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Sidebar Menu
st.sidebar.title("🌿 Plant Disease Detection")
app_mode = st.sidebar.radio("📌 Select Page:", ["🏠 Home", "🔬 Disease Recognition"])

# Display banner image
try:
    img = Image.open("diseases.jpg")
    st.image(img, use_container_width=True)
except FileNotFoundError:
    st.warning("⚠️ Banner image 'diseases.jpg' not found. Please check the file path.")

# Homepage Content
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center; font-size:40px;'>🌱 Plant Disease Detection System 🌱</h1>", unsafe_allow_html=True)
    
    st.write("""
        This AI-powered system helps farmers and researchers **detect plant diseases** using deep learning.  
        Simply upload a **leaf image**, and our trained model will classify it as **healthy** or **infected**.

        **🌟 Features:**
        - **Fast & Accurate** disease prediction  
        - **Easy to Use** - Just upload an image!  
        - **Supports Potato Disease Detection**  
    """)

    st.subheader("💡 How It Works:")
    st.write("""
    1️⃣ Upload a **clear image** of the plant leaf.  
    2️⃣ Click on **Predict**, and our model will analyze it.  
    3️⃣ Get instant results with **disease classification**!  
    """)

    st.write("➡️ **Go to the 'Disease Recognition' tab in the sidebar to start detecting plant diseases!**")

# Disease Recognition Page
elif app_mode == "🔬 Disease Recognition":
    st.header("🔬 Plant Disease Recognition System")
    
    # File uploader
    test_image = st.file_uploader("📸 Choose an Image:", type=["jpg", "png", "jpeg"])

    # Layout buttons in two columns
    col1, col2 = st.columns([1, 1])

    # Show uploaded image only if test_image is not None
    with col1:
        if st.button("📷 Show Image"):
            if test_image is not None:
                st.image(test_image, use_container_width=True)
            else:
                st.warning("⚠️ Please upload an image first.")

    # Define information for each disease category
    disease_info = {
        "Potato___Early_blight": """🛑 **Early Blight**  
        - **Cause**: Fungus *Alternaria solani*  
        - **Symptoms**: Dark brown spots with concentric rings on leaves.  
        - **Prevention**: Avoid overhead watering, use resistant varieties, and apply fungicides.  
        """,
        
        "Potato___Late_blight": """⚠️ **Late Blight**  
        - **Cause**: Pathogen *Phytophthora infestans*  
        - **Symptoms**: Dark, water-soaked lesions that spread rapidly in humid conditions.  
        - **Prevention**: Remove infected plants, improve air circulation, and use fungicides.  
        """,

        "Potato___healthy": """✅ **Healthy Plant**  
        - Your plant appears **healthy** with no visible disease symptoms. 🎉  
        - Keep monitoring for any changes and maintain good farming practices!  
        """
    }

    # Predict button only if test_image is not None
    with col2:
        if st.button("🔍 Predict"):
            if test_image is not None:
                st.balloons()
                st.write("✅ **Our Prediction**")
                result_index = model_prediction(test_image)
                class_name = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
                predicted_category = class_name[result_index]

                # Display the prediction result with additional information
                st.success(f"🌿 **Model predicts:** {predicted_category}")
                st.write(disease_info[predicted_category])  # Show relevant disease details
            else:
                st.warning("⚠️ Please upload an image before predicting.")

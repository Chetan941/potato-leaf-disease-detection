import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function for model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# 🎨 Custom CSS for a bright and colorful background
st.markdown(
    """
    <style>
    /* Full-page vibrant gradient background */
    body, .stApp {
        background: linear-gradient(to right, #6a11cb, #2575fc) !important;
        color: white !important;
    }

    /* Sidebar with a fresh gradient */
    .stSidebar {
        background: linear-gradient(to bottom, #11998e, #38ef7d) !important;
        color: white !important;
    }

    /* Sidebar text */
    .stSidebar .css-1d391kg {
        color: white !important;
    }

    /* TOP HEADER: Bold & colorful */
    .header-style {
        background: linear-gradient(to right, #ff512f, #dd2476);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
    }

    /* Buttons: Bright Gold */
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        font-size: 18px;
        border-radius: 12px;
        padding: 12px 20px;
    }

    .stButton>button:hover {
        background-color: #FFA500;
        color: #fff;
    }

    /* Radio button styling */
    div[data-baseweb="radio"] label {
        font-size: 20px !important;
        font-weight: bold !important;
        color: #FFD700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menu with new colors
st.sidebar.title("🌿 **Plant Disease Detection**")
app_mode = st.sidebar.radio("📌 **Select The Page**", ["Homepage", "Disease Recognition"])

# Display banner image (Ensure 'diseases.jpg' exists in the project directory)
try:
    img = Image.open("diseases.jpg")
    st.image(img, use_container_width=True)
except FileNotFoundError:
    st.warning("⚠️ Banner image 'diseases.jpg' not found. Please check the file path.")

# Homepage content
if app_mode == "Homepage":
    st.markdown("<div class='header-style'>🌱 Welcome to Plant Disease Detection 🌱</div>", unsafe_allow_html=True)
    
    # Add project details
    st.write("""
        This AI-based project helps farmers and researchers **detect plant diseases** using deep learning.  
        Simply upload a **leaf image**, and our **trained model** will classify it as **healthy** or **infected** with diseases like **Early Blight** or **Late Blight**.

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

    # Encourage users to navigate using the sidebar
    st.write("➡️ **Go to the 'Disease Recognition' tab in the sidebar to start detecting plant diseases!**")

# Disease Recognition page
elif app_mode == "Disease Recognition":
    st.header("🔬 Plant Disease Recognition System")

    # File uploader
    test_image = st.file_uploader("📸 Choose an Image:")

    # Layout buttons in two columns
    col1, col2 = st.columns([1, 1])

    # Show uploaded image only if test_image is not None
    with col1:
        if st.button("📷 Show The Image"):
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

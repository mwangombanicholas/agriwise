import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide")

# --- MODEL DOWNLOADER ---
MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model for the first time (61MB). Please wait..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🌾 AgriWise")
st.markdown("### AI-Powered Crop Health Advisor")
st.markdown("**Supported crop:** 🌽 Maize (Corn) | **Model Accuracy:** 96%")

tab1, tab2 = st.tabs(["📸 Disease Detection", "⛅ Weather Advisory"])

with tab1:
    st.header("Upload a photo of your maize leaf")
    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)
        
        # 1. THE BOUNCER: Check if it's actually a plant
        img_array = np.array(image.resize((128, 128)))
        avg_color = np.mean(img_array, axis=(0, 1)) 
        
        # Reject dark images, people, and documents
        is_plant = (avg_color[1] > 50) and (avg_color[1] > avg_color[0] * 0.8) and (avg_color[1] > avg_color[2] * 0.8)
        
        if not is_plant:
            st.error("❌ **Invalid Image Detected.**")
            st.info("This does not look like a plant leaf. Please upload a clear, close-up photo of a maize leaf.")
        
        else:
            # 2. THE AI MODEL
            try:
                with st.spinner("Loading AI Model... (If this is the first time, it may take 1 minute)"):
                    model = load_model()
                
                with st.spinner("Analyzing..."):
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = model.predict(img_array)
                    idx = np.argmax(predictions[0])
                    confidence = predictions[0][idx] * 100
                    
                    class_names = ['maize_cercospora_leaf_spot', 'maize_common_rust', 'maize_healthy', 'maize_northern_leaf_blight']
                    display_names = {
                        'maize_cercospora_leaf_spot': 'Maize Cercospora Leaf Spot',
                        'maize_common_rust': 'Maize Common Rust',
                        'maize_healthy': 'Healthy Maize',
                        'maize_northern_leaf_blight': 'Maize Northern Leaf Blight'
                    }
                    result = class_names[idx]
                    display = display_names[result]
                    
                    # 3. CONFIDENCE CHECK
                    if confidence < 60.0:
                        st.warning("⚠️ **Unable to make a reliable diagnosis.**")
                        st.info("The image might be blurry, too far away, or not a clear maize leaf. Please try again.")
                    else:
                        if 'healthy' in result:
                            st.success(f"### ✅ Diagnosis: {display}")
                        else:
                            st.warning(f"### ⚠️ Diagnosis: {display}")
                        st.info(f"**Confidence:** {confidence:.1f}%")
                        
                        treatment_info = {
                            'healthy': {'chemical': 'No treatment needed', 'organic': 'Continue good practices'},
                            'common_rust': {'chemical': 'Azoxystrobin fungicide', 'organic': 'Neem oil spray'},
                            'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide', 'organic': 'Remove infected leaves'},
                            'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution'}
                        }
                        key = result.replace('maize_', '')
                        info = treatment_info.get(key, treatment_info['healthy'])
                        
                        st.subheader("💊 Treatment")
                        col1, col2 = st.columns(2)
                        col1.write(f"**Chemical:** {info['chemical']}")
                        col2.write(f"**Organic:** {info['organic']}")
            except Exception as e:
                st.error("Failed to load the AI model. Please refresh the page and try again.")

with tab2:
    st.header("Weather Advisory")
    district = st.selectbox("Select district", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
    weather_alerts = {
        "Mzuzu": ("No active alerts", "success"),
        "Mzimba": ("🌵 DRY SPELL ALERT: Irrigate if possible", "warning"),
        "Karonga": ("🚨 FLOOD ALERT: Heavy rains expected", "error"),
        "Rumphi": ("No active alerts", "success"),
        "Nkhata Bay": ("🚨 FLOOD ALERT: Heavy rains expected", "error")
    }
    msg, alert_type = weather_alerts.get(district, ("No active alerts", "success"))
    if alert_type == "error": st.error(msg)
    elif alert_type == "warning": st.warning(msg)
    else: st.success(msg)
    st.info("📅 Maize planting: November-December")

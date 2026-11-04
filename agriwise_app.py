import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide")

MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

def is_it_a_leaf(image):
    img = image.resize((128, 128))
    img_rgb = np.array(img)
    img_hsv = np.array(img.convert('HSV'))
    hue = img_hsv[:, :, 0]
    sat = img_hsv[:, :, 1] 
    plant_pixels = np.sum(((hue >= 15) & (hue <= 100)) & (sat > 40))
    total_pixels = 128 * 128
    color_score = (plant_pixels / total_pixels) * 100
    if color_score < 25: return False
    gray = np.mean(img_rgb, axis=2)
    edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    texture_score = edges / total_pixels
    if texture_score < 10: return False
    return True

st.title("🌾 AgriWise")
st.markdown("### AI-Powered Crop Health Advisor")
st.markdown("**Supported crop:** 🌽 Maize (Corn)")

tab1, tab2 = st.tabs(["📸 Disease Detection", "⛅ Weather Advisory"])

# ================= TAB 1: ORIGINAL LAYOUT =================
with tab1:
    st.header("Upload a photo of your maize leaf")
    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    st.caption("⚠️ Note: This AI is trained for maize leaves. Photos of people or objects will be rejected.")
    
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)
        
        if not is_it_a_leaf(image):
            st.error("❌ **Invalid Image Detected.**")
            st.info("This does not look like a plant leaf. Please take a clear photo of a maize leaf.")
        else:
            try:
                model = load_model()
                with st.spinner("Analyzing crop..."):
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
                st.error(f"Error analyzing image: {str(e)}")

# ================= TAB 2: VIDEO BACKGROUND =================
with tab2:
    # 1. Read the video file
    video_file = "weather_bg.mp4"
    if os.path.exists(video_file):
        with open(video_file, "rb") as f:
            video_bytes = f.read()
            encoded_video = base64.b64encode(video_bytes).decode()
        
        # 2. HTML/CSS for the video background
        video_html = f"""
        <style>
        .video-container {{
            position: relative;
            width: 100%;
            height: 400px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}
        .video-bg {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: 0;
            opacity: 0.8;
        }}
        .content-overlay {{
            position: relative;
            z-index: 1;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            margin: 2rem;
            max-width: 500px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        </style>
        
        <div class="video-container">
            <video autoplay loop muted playsinline class="video-bg">
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
            </video>
            
            <div class="content-overlay">
                <h2>⛅ Weather Advisory</h2>
                <p>Select your district below to view local forecasts and agricultural alerts.</p>
            </div>
        </div>
        """
        components.html(video_html, height=420)
    
    # 3. The actual weather logic (below the video)
    st.header("Select District")
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

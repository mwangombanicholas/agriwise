import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

# --- LOAD BACKGROUND IMAGE AS BASE64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

bg_img = get_base64_of_bin_file('confield.jpg')

# --- CUSTOM CSS FOR THE UI ---
st.markdown(f"""
<style>
    /* Background Image */
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Main Title */
    .main-title {{
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }}
    
    /* Hide standard Streamlit header */
    header {{visibility: hidden;}}
    
    /* Style the uploader box to look dark */
    [data-testid="stFileUploader"] {{
        background-color: rgba(30, 30, 30, 0.8);
        border-radius: 10px;
        padding: 10px;
    }}
    
    /* Tab text color */
    .stTabs [data-baseweb="tab"] {{
        color: white;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<p class="main-title">Bringing Innovation to Your Farming Journey.</p>', unsafe_allow_html=True)
st.write("") # Spacing

# --- MODEL LOADING ---
MODEL_PATH = "agriwise_model.h5"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

# --- BOUNCER LOGIC ---
def is_it_a_leaf(image):
    img = image.resize((128, 128))
    img_rgb = np.array(img)
    
    img_hsv = np.array(img.convert('HSV'))
    hue = img_hsv[:, :, 0]
    sat = img_hsv[:, :, 1] 
    
    plant_pixels = np.sum(((hue >= 15) & (hue <= 100)) & (sat > 40))
    total_pixels = 128 * 128
    color_score = (plant_pixels / total_pixels) * 100
    
    if color_score < 25:
        return False
        
    gray = np.mean(img_rgb, axis=2)
    edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    texture_score = edges / total_pixels
    
    if texture_score < 10:
        return False
        
    return True

# --- TABS ---
tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: white; border-radius: 30px; padding: 10px 20px; margin-bottom: 10px; color: #2e7d32; font-weight: bold; display: flex; align-items: center;">
            🌱 Better Growth
        </div>
        <div style="background-color: white; border-radius: 30px; padding: 10px 20px; margin-bottom: 10px; color: #c62828; font-weight: bold; display: flex; align-items: center;">
            🛡️ Strong Protection
        </div>
        <div style="background-color: white; border-radius: 30px; padding: 10px 20px; margin-bottom: 10px; color: #1565c0; font-weight: bold; display: flex; align-items: center;">
            📈 Higher Yield
        </div>
        <div style="background-color: white; border-radius: 30px; padding: 10px 20px; margin-bottom: 10px; color: #2e7d32; font-weight: bold; display: flex; align-items: center;">
            🌍 Sustainable Future
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='color: white; text-shadow: 1px 1px 4px black;'>📸 Scan Your Maize Crop</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            
            if not is_it_a_leaf(image):
                st.error("❌ **Invalid Image Detected.** Please upload a clear photo of a maize leaf.")
            else:
                model = load_model()
                if model:
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
                        
                        st.success(f"**Diagnosis: {display}**")
                        st.info(f"Confidence: {confidence:.1f}%")
                        
                        treatment_info = {
                            'healthy': {'chemical': 'No treatment needed', 'organic': 'Continue good practices'},
                            'common_rust': {'chemical': 'Azoxystrobin fungicide', 'organic': 'Neem oil spray'},
                            'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide', 'organic': 'Remove infected leaves'},
                            'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution'}
                        }
                        key = result.replace('maize_', '')
                        info = treatment_info.get(key, treatment_info['healthy'])
                        
                        st.markdown("**💊 Treatment:**")
                        st.write(f"- Chemical: {info['chemical']}")
                        st.write(f"- Organic: {info['organic']}")
                else:
                    st.error("Model not found. Please ensure agriwise_model.h5 is uploaded.")

with tab2:
    # --- WEATHER BACKGROUND VIDEO ---
    video_file = "weather_bg.mp4"
    if os.path.exists(video_file):
        with open(video_file, "rb") as video:
            video_bytes = video.read()
        st.video(video_bytes, muted=True, autoplay=True, loop=True)
    # --------------------------------
    
    st.markdown("<h3 style='color: white; text-shadow: 1px 1px 4px black;'>⛅ Localized Climate Advisory</h3>", unsafe_allow_html=True)
    district = st.selectbox("📍 Select District", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
    
    weather_alerts = {
        "Mzuzu": ("No active weather alerts for your area.", "success"),
        "Mzimba": ("🌵 DRY SPELL ALERT: Irrigate if possible", "warning"),
        "Karonga": ("🚨 FLOOD ALERT: Heavy rains expected", "error"),
        "Rumphi": ("No active weather alerts for your area.", "success"),
        "Nkhata Bay": ("🚨 FLOOD ALERT: Heavy rains expected", "error")
    }
    msg, alert_type = weather_alerts.get(district, ("No active weather alerts for your area.", "success"))
    
    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("🌡️ Temperature", "24°C")
    col_w2.metric("🌧️ Rainfall", "Moderate")
    col_w3.metric("🗓️ Season", "Rainy / Growing")
    
    if alert_type == "error": st.error(msg)
    elif alert_type == "warning": st.warning(msg)
    else: st.success(msg)
    
    st.markdown("""
    <div style="background-color: rgba(30,30,30,0.7); padding: 15px; border-radius: 10px; color: white;">
        <b>🗓️ Planting Calendar:</b><br>
        • 🌽 Maize: Plant in November - December<br>
        • 🌾 Harvest: April - June
    </div>
    """, unsafe_allow_html=True)

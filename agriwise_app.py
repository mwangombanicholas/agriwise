import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide")

MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

# --- BACKGROUND IMAGE SETUP ---
bg_image_path = "cornfield.jpg"
if os.path.exists(bg_image_path):
    with open(bg_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_url = f"data:image/jpeg;base64,{encoded_string}"
else:
    bg_url = "https://images.unsplash.com/photo-1592982537447-7440770cbfc9?q=80&w=2070&auto=format&fit=crop"

# --- CUSTOM CSS TO FIX THE TEXT AND LAYOUT ---
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Glassmorphism container for the main content */
    .block-container {{
        background: rgba(255, 255, 255, 0.92); /* Made slightly more opaque for readability */
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}
    
    /* Force ALL text to be dark so it's readable on the white background */
    h1, h2, h3, h4, p, span, label, div {{
        color: #1b5e20 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    
    /* Fix the File Uploader Box */
    [data-testid="stFileUploader"] {{
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #2e7d32;
    }}
    
    /* Professional Buttons */
    .stButton>button {{
        background-color: #2e7d32;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(5px);
    }}
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🌱 About AgriWise")
    st.write("Built by **Mzuzu University Students** for smallholder farmers.")
    st.markdown("---")
    st.markdown("**Supported Crop:** 🌽 Maize (Corn)")
    st.markdown("**Model Accuracy:** 96%")
    st.markdown("---")
    st.markdown("#### 🔍 Detects:")
    st.markdown("- Healthy Maize\n- Common Rust\n- Cercospora Leaf Spot\n- Northern Leaf Blight")

# --- MAIN UI ---
st.title("🌾 AgriWise Dashboard")
st.markdown("#### AI-Powered Crop Health & Climate Advisor")
st.markdown("---")

tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])

with tab1:
    st.header("Scan Your Maize Crop")
    st.markdown("Take a clear, close-up photo of a single maize leaf.")
    
    uploaded = st.file_uploader("Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    st.caption("⚠️ Note: This AI is trained for maize leaves. Photos of people or objects will be rejected.")
    
    if uploaded:
        col1, col2 = st.columns([1, 2])
        image = Image.open(uploaded).convert('RGB')
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        with col2:
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
                        
                        st.subheader("💊 Treatment Recommendations")
                        t1, t2 = st.columns(2)
                        t1.info(f"**🧪 Chemical:**\n\n{info['chemical']}")
                        t2.success(f"**🌿 Organic:**\n\n{info['organic']}")
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")

with tab2:
    st.header("Localized Climate Advisory")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        district = st.selectbox("📍 Select District", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
    
    weather_alerts = {
        "Mzuzu": ("No active weather alerts for your area.", "success"),
        "Mzimba": ("🌵 DRY SPELL ALERT: Irrigate if possible", "warning"),
        "Karonga": ("🚨 FLOOD ALERT: Heavy rains expected", "error"),
        "Rumphi": ("No active weather alerts for your area.", "success"),
        "Nkhata Bay": ("🚨 FLOOD ALERT: Heavy rains expected", "error")
    }
    
    msg, alert_type = weather_alerts.get(district, ("No active weather alerts for your area.", "success"))
    
    st.markdown("---")
    
    # Weather Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("🌡️ Temperature", "24°C")
    m2.metric("🌧️ Rainfall", "Moderate")
    m3.metric("📅 Season", "Rainy / Growing")

    if alert_type == "error": st.error(msg)
    elif alert_type == "warning": st.warning(msg)
    else: st.success(msg)
    
    st.info("📅 **Planting Calendar:**\n- **Maize:** Plant in November - December\n- **Harvest:** April - June")

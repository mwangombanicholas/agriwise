import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="AgriWise | Smart Farming", page_icon="🌽", layout="wide", initial_sidebar_state="collapsed")

# ==================== BACKGROUND IMAGE SETUP ====================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_img_base64 = get_base64_of_bin_file('cornfield.jpg')
    bg_css = f'background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)), url("data:image/jpeg;base64,{bg_img_base64}");'
except:
    bg_css = 'background-color: #1B5E20;'

st.markdown(f"""
<style>
    /* The Background Image */
    .stApp {{
        {bg_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Hide default Streamlit headers */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Global Text Colors (White for the dark background) */
    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: white !important;
    }}
    
    /* Tab Styling to make them visible */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.2rem;
        font-weight: bold;
        color: white !important;
    }}
    
    /* Hero Section */
    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    
    /* White Pill Cards */
    .pill-card {{
        background-color: white;
        border-radius: 50px;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        max-width: 300px;
    }}
    .pill-card p {{
        color: #1B5E20 !important; /* Force dark green inside white box */
        font-weight: bold;
        margin: 0;
        margin-left: 15px;
        font-size: 1.1rem;
    }}
    
    /* Upload Box Styling */
    .upload-container {{
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }}
    
    /* Force text inside upload container to be dark */
    .upload-container h3, .upload-container p {{
        color: #1B5E20 !important;
    }}
    
    /* Result Card */
    .result-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }}
    
    /* Force text inside result card to be dark */
    .result-card h3, .result-card p, .result-card h4 {{
        color: #1B5E20 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADER ====================
MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🌱 Initializing AI Model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

# ==================== HEADER ====================
st.markdown('<h1 class="hero-title">Bringing Innovation to Your Farming Journey.</h1>', unsafe_allow_html=True)

# ==================== MAIN TABS ====================
# Bringing back the tabs so Weather is visible!
tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("""
        <div class="pill-card"><span>🌱</span><p>Better Growth</p></div>
        <div class="pill-card"><span>🛡️</span><p>Strong Protection</p></div>
        <div class="pill-card"><span>📈</span><p>Higher Yield</p></div>
        <div class="pill-card"><span>🌍</span><p>Sustainable Future</p></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0;'>📸 Scan Your Maize Crop</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a clear photo of a maize leaf", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== PROCESSING ====================
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.image(image, caption="Your Crop", use_column_width=True)
            
        with res_col2:
            try:
                model = load_model()
                with st.spinner("🧠 AI is analyzing the leaf..."):
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
                        border_color = "#2E7D32"
                        icon = "✅"
                        msg = "Your crop is in good health! Keep up the good farming practices."
                    elif confidence < 60:
                        border_color = "#C62828"
                        icon = "❌"
                        msg = "The AI could not confidently identify a maize leaf here. Please ensure you uploaded a clear photo."
                        display = "Unrecognized Image"
                    else:
                        border_color = "#EF6C00"
                        icon = "⚠️"
                        msg = "Disease detected. Follow the treatment plan below immediately."

                    st.markdown(f"""
                    <div class="result-card" style="border-left: 8px solid {border_color};">
                        <h3 style="margin-top:0;">{icon} {display}</h3>
                        <p style="font-size:1.2rem;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                        <p>{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'healthy' not in result and confidence >= 60:
                        treatment_info = {
                            'common_rust': {'chemical': 'Azoxystrobin fungicide (K8,000/ha)', 'organic': 'Neem oil spray (2 tsp/L)'},
                            'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide (K5,000/kg)', 'organic': 'Remove infected leaves. Copper spray.'},
                            'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution (1 tbsp/L)'}
                        }
                        key = result.replace('maize_', '')
                        info = treatment_info.get(key, {'chemical': 'Consult expert', 'organic': 'Consult expert'})
                        
                        st.markdown(f"""
                        <div style="display: flex; gap: 15px; margin-top: 15px;">
                            <div style="background:white; padding:20px; border-radius:15px; flex:1; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                                <h4 style="margin-top:0;">🧪 Chemical</h4>
                                <p style="font-size:1rem;">{info['chemical']}</p>
                            </div>
                            <div style="background:white; padding:20px; border-radius:15px; flex:1; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                                <h4 style="margin-top:0;">🌿 Organic</h4>
                                <p style="font-size:1rem;">{info['organic']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== TAB 2: WEATHER ====================
with tab2:
    st.markdown("### ⛅ Localized Climate Advisory")
    
    districts = {
        "Mzuzu": {"temp": "24°C", "rain": "Moderate", "alert": "none"},
        "Mzimba": {"temp": "26°C", "rain": "Low", "alert": "dry"},
        "Karonga": {"temp": "29°C", "rain": "High", "alert": "flood"},
        "Rumphi": {"temp": "23°C", "rain": "Moderate", "alert": "none"},
        "Nkhata Bay": {"temp": "27°C", "rain": "Very High", "alert": "flood"}
    }
    
    selected = st.selectbox("📍 Select District", list(districts.keys()))
    
    if selected:
        data = districts[selected]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="🌡️ Temperature", value=data["temp"])
        with m2:
            st.metric(label="🌧️ Rainfall", value=data["rain"])
        with m3:
            st.metric(label="📅 Season", value="Rainy / Growing" if data["alert"] != "dry" else "Dry")
            
        if data["alert"] == "flood":
            st.error("🚨 **FLOOD ALERT:** Heavy rains expected in the next 48 hours. Move livestock to higher ground.")
        elif data["alert"] == "dry":
            st.warning("🌵 **DRY SPELL ALERT:** Low rainfall expected for the next 7 days. Irrigate crops if possible.")
        else:
            st.success("✅ **No active weather alerts** for your area.")
            
        st.info("""
        **📅 Planting Calendar:**
        * **🌽 Maize:** Plant in November - December
        * **🌾 Harvest:** April - June
        """)

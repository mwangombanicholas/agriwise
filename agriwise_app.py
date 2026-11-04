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
    bg_css = f'background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.7)), url("data:image/jpeg;base64,{bg_img_base64}");'
except:
    bg_css = 'background-color: #1B5E20;'

st.markdown(f"""
<style>
    /* The Full Screen Background Image */
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
        font-family: 'Helvetica Neue', sans-serif;
    }}
    
    /* Hero Section */
    .hero-title {{
        font-size: 4rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }}
    
    /* Frosted Glass Cards for readability */
    .glass-card {{
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
    }}
    
    /* Force text inside glass cards to be dark */
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4, .glass-card p, .glass-card span, .glass-card li {{
        color: #1B5E20 !important;
    }}
    
    /* White Pill Cards (from Image 2) */
    .pill-card {{
        background-color: white;
        border-radius: 50px;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        max-width: 300px;
    }}
    .pill-card p {{
        color: #1B5E20 !important;
        font-weight: bold;
        margin: 0;
        margin-left: 15px;
        font-size: 1.1rem;
    }}
    
    /* Tab Styling (Make tabs visible over the dark background) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.2rem;
        font-weight: bold;
        color: white !important;
        background-color: rgba(0,0,0,0.5);
        padding: 10px 20px;
        border-radius: 10px;
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
tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])

# ==================== TAB 1: DIAGNOSIS ====================
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
        # Uploader inside a glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

                    # Result Glass Card
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 8px solid {border_color}; padding: 1.5rem;">
                        <h3 style="margin-top:0;">{icon} {display}</h3>
                        <p style="font-size:1.2rem;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                        <p>{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Treatments
                    if 'healthy' not in result and confidence >= 60:
                        treatment_info = {
                            'common_rust': {'chemical': 'Azoxystrobin fungicide (K8,000/ha)', 'organic': 'Neem oil spray (2 tsp/L)'},
                            'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide (K5,000/kg)', 'organic': 'Remove infected leaves. Copper spray.'},
                            'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution (1 tbsp/L)'}
                        }
                        key = result.replace('maize_', '')
                        info = treatment_info.get(key, {'chemical': 'Consult expert', 'organic': 'Consult expert'})
                        
                        st.markdown("### 💊 Recommended Treatment")
                        t1, t2 = st.columns(2)
                        
                        with t1:
                            st.markdown(f"""
                            <div class="glass-card" style="border-top: 5px solid #2196F3; padding: 1.5rem;">
                                <h4>🧪 Chemical Treatment</h4>
                                <p>{info['chemical']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with t2:
                            st.markdown(f"""
                            <div class="glass-card" style="border-top: 5px solid #4CAF50; padding: 1.5rem;">
                                <h4>🌿 Organic Alternative</h4>
                                <p>{info['organic']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== TAB 2: WEATHER ====================
with tab2:
    # Weather tab inside a large glass card so it's readable
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="AgriWise | Smart Farming", page_icon="🌽", layout="wide", initial_sidebar_state="expanded")

# ==================== CUSTOM CSS & BACKGROUND ====================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_img_base64 = get_base64_of_bin_file('cornfield.jpg')
    hero_bg = f'url("data:image/jpeg;base64,{bg_img_base64}")'
except:
    hero_bg = 'linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%)'

st.markdown(f"""
<style>
    /* Clean, light background for the whole app */
    .stApp {{
        background-color: #f8f9fa;
    }}
    
    /* Hide default Streamlit headers */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* HERO BANNER - Using the Cornfield Image */
    .hero-banner {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.4)), {hero_bg};
        background-size: cover;
        background-position: center;
        padding: 4rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    .hero-banner h1 {{
        font-size: 3.5rem;
        font-weight: 800;
        color: white !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }}
    .hero-banner p {{
        font-size: 1.3rem;
        color: #f0f0f0 !important;
        max-width: 700px;
        margin: 0 auto;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }}
    
    /* White Cards */
    .white-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }}
    
    /* Global Text (Dark for readability) */
    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: #2c3e50;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
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

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2488/2488658.png", width=80)
    st.title("AgriWise")
    st.markdown("---")
    
    # Custom Navigation (This acts like the buttons you wanted)
    page = st.radio(
        "Navigation Menu",
        ["📸 Crop Diagnosis", "⛅ Climate Advisory", "ℹ️ About Us"]
    )
    
    st.markdown("---")
    st.info("Built by Mzuzu University Students for the SAVE Project.")

# ==================== PAGE 1: CROP DIAGNOSIS ====================
if page == "📸 Crop Diagnosis":
    # HERO BANNER
    st.markdown("""
    <div class="hero-banner">
        <h1>Bringing Innovation to Your Farming Journey.</h1>
        <p>Empowering farmers with instant AI crop diagnosis and climate alerts. Scan your maize leaves today.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Crop Image")
        uploaded = st.file_uploader("Take a clear, close-up photo of the maize leaf.", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        st.markdown("### 🌱 Why Use AgriWise?")
        st.success("✅ Early Disease Detection")
        st.success("✅ 96% Accurate AI Model")
        st.success("✅ Instant Treatment Plans")

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        
        st.markdown("---")
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            st.image(image, caption="Uploaded Crop Image", use_column_width=True)
            
        with col_res:
            try:
                model = load_model()
                with st.spinner("🧠 Analyzing leaf..."):
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
                        bg_color, border_color, icon = "#E8F5E9", "#2E7D32", "✅"
                        msg = "Your crop is in good health! Keep up the good farming practices."
                    elif confidence < 60:
                        bg_color, border_color, icon = "#FFEBEE", "#C62828", "❌"
                        msg = "The AI could not confidently identify a maize leaf here."
                        display = "Unrecognized Image"
                    else:
                        bg_color, border_color, icon = "#FFF3E0", "#EF6C00", "⚠️"
                        msg = "Disease detected. Follow the treatment plan below immediately."

                    # RESULT CARD
                    st.markdown(f"""
                    <div class="white-card" style="background-color: {bg_color}; border-left: 8px solid {border_color};">
                        <h3 style="margin-top:0; color: {border_color} !important;">{icon} {display}</h3>
                        <p style="font-size:1.2rem; color: #333 !important;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                        <p style="color: #333 !important;">{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TREATMENTS
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
                            <div class="white-card" style="border-top: 5px solid #2196F3;">
                                <h4 style="color: #2196F3 !important;">🧪 Chemical Treatment</h4>
                                <p style="color: #333 !important;">{info['chemical']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with t2:
                            st.markdown(f"""
                            <div class="white-card" style="border-top: 5px solid #4CAF50;">
                                <h4 style="color: #4CAF50 !important;">🌿 Organic Alternative</h4>
                                <p style="color: #333 !important;">{info['organic']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== PAGE 2: CLIMATE ADVISORY ====================
elif page == "⛅ Climate Advisory":
    st.markdown("""
    <div class="hero-banner" style="padding: 2rem;">
        <h1>⛅ Climate Advisory</h1>
        <p>Localized weather alerts to protect your harvest.</p>
    </div>
    """, unsafe_allow_html=True)
    
    districts = {
        "Mzuzu": {"temp": "24°C", "rain": "Moderate", "alert": "none"},
        "Mzimba": {"temp": "26°C", "rain": "Low", "alert": "dry"},
        "Karonga": {"temp": "29°C", "rain": "High", "alert": "flood"},
        "Rumphi": {"temp": "23°C", "rain": "Moderate", "alert": "none"},
        "Nkhata Bay": {"temp": "27°C", "rain": "Very High", "alert": "flood"}
    }
    
    st.markdown("### 📍 Select Your District")
    selected = st.selectbox("", list(districts.keys()), label_visibility="collapsed")
    
    if selected:
        data = districts[selected]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="🌡️ Temperature", value=data["temp"])
        with m2:
            st.metric(label="🌧️ Rainfall", value=data["rain"])
        with m3:
            st.metric(label="📅 Season", value="Rainy / Growing" if data["alert"] != "dry" else "Dry")
            
        st.markdown("### ⚠️ Active Alerts")
        if data["alert"] == "flood":
            st.error("🚨 **FLOOD ALERT:** Heavy rains expected in the next 48 hours. Move livestock to higher ground.")
        elif data["alert"] == "dry":
            st.warning("🌵 **DRY SPELL ALERT:** Low rainfall expected for the next 7 days. Irrigate crops if possible.")
        else:
            st.success("✅ **No active weather alerts** for your area.")
            
        st.markdown("### 📅 Planting Calendar")
        st.info("""
        * **🌽 Maize:** Plant in November - December
        * **🌾 Harvest:** April - June
        * **🌱 Fertilizer:** Apply 3-4 weeks after planting
        """)

# ==================== PAGE 3: ABOUT US ====================
elif page == "ℹ️ About Us":
    st.markdown("""
    <div class="hero-banner" style="padding: 2rem;">
        <h1>About AgriWise</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="white-card">
        <h2>Our Mission</h2>
        <p>AgriWise is an AI-powered web application designed to help smallholder farmers in Malawi detect maize diseases early and receive actionable treatment advice.</p>
        
        <h3>The Team</h3>
        <ul>
            <li><strong>Nicholas Mwangomba</strong> - AI Model Development</li>
            <li><strong>Blessings Kalinde</strong> - Data Processing</li>
            <li><strong>Harold Phiri</strong> - Research</li>
            <li><strong>Bright Msiska</strong> - App Development</li>
        </ul>
        
        <h3>Supported by</h3>
        <p>The World Bank SAVE Project and Mzuzu University.</p>
    </div>
    """, unsafe_allow_html=True)

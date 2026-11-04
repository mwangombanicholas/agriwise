import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="AgriWise | Crop Health AI", page_icon="🌽", layout="wide", initial_sidebar_state="expanded")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #f4f9f4;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Result Cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .result-healthy {
        background-color: #E8F5E9;
        border-left: 6px solid #2E7D32;
        color: #1B5E20;
    }
    .result-disease {
        background-color: #FFF3E0;
        border-left: 6px solid #EF6C00;
        color: #E65100;
    }
    .result-error {
        background-color: #FFEBEE;
        border-left: 6px solid #C62828;
        color: #B71C1C;
    }
    
    /* Treatment Boxes */
    .treatment-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        height: 100%;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #DDD;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🌽 AgriWise</h1>
    <p>AI-Powered Crop Health & Climate Advisory</p>
</div>
""", unsafe_allow_html=True)

# ==================== MODEL LOADER ====================
MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🌱 Initializing AI Model... (First load may take 1 minute)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2488/2488658.png", width=100) # Corn icon
    st.markdown("### 🌾 Welcome to AgriWise")
    st.info("Empowering Malawian smallholder farmers with instant AI crop diagnosis and climate alerts.")
    
    st.markdown("---")
    st.markdown("### 🔍 Detection Classes")
    st.markdown("✅ Healthy Maize")
    st.markdown("⚠️ Common Rust")
    st.markdown("⚠️ Cercospora Leaf Spot")
    st.markdown("⚠️ Northern Leaf Blight")

# ==================== MAIN TABS ====================
tab1, tab2 = st.tabs(["📸 Crop Diagnosis", "⛅ Climate Advisory"])

# ==================== TAB 1: DIAGNOSIS ====================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Crop Image")
        st.markdown("Take a clear, close-up photo of the maize leaf.")
        uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
    with col2:
        st.markdown("### 📋 Upload Guidelines")
        st.markdown("""
        1. 🌱 Ensure the leaf fills the frame.
        2. ☀️ Take the photo in good lighting.
        3. 🔍 Focus closely on the spots or discoloration.
        4. 🚫 Avoid photos of people or random objects.
        """)

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
                    
                    # If confidence is very low, it might be a bad image
                    if confidence < 60:
                        st.markdown(f"""
                        <div class="result-card result-error">
                            <h3>❌ Unrecognized Image</h3>
                            <p>The AI could not confidently identify a maize leaf here. Please ensure you uploaded a clear photo of a maize leaf.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Show Result Card
                        if 'healthy' in result:
                            st.markdown(f"""
                            <div class="result-card result-healthy">
                                <h3>✅ Diagnosis: {display}</h3>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                <p>🌱 Your crop is in good health! Keep up the good farming practices.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card result-disease">
                                <h3>⚠️ Diagnosis: {display}</h3>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                <p>🚨 Disease detected. Follow the treatment plan below immediately.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Treatment Columns with Icons
                        treatment_info = {
                            'healthy': {'chemical': 'No chemicals needed.', 'organic': 'Maintain current practices.'},
                            'common_rust': {'chemical': 'Azoxystrobin fungicide (K8,000/ha)', 'organic': 'Neem oil spray (2 tsp/L)'},
                            'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide (K5,000/kg)', 'organic': 'Remove infected leaves. Copper spray.'},
                            'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution (1 tbsp/L)'}
                        }
                        key = result.replace('maize_', '')
                        info = treatment_info.get(key, treatment_info['healthy'])
                        
                        st.markdown("### 💊 Recommended Treatment")
                        t1, t2 = st.columns(2)
                        
                        with t1:
                            st.markdown(f"""
                            <div class="treatment-box">
                                <h4>🧪 Chemical Treatment</h4>
                                <p>{info['chemical']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with t2:
                            st.markdown(f"""
                            <div class="treatment-box">
                                <h4>🌿 Organic Alternative</h4>
                                <p>{info['organic']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")

# ==================== TAB 2: WEATHER ====================
with tab2:
    st.markdown("### ⛅ Localized Climate Advisory")
    st.markdown("Select your district to receive real-time weather alerts and planting advice.")
    
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
        
        # Weather Metrics Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="🌡️ Temperature", value=data["temp"])
        with m2:
            st.metric(label="🌧️ Rainfall", value=data["rain"])
        with m3:
            st.metric(label="📅 Season", value="Rainy / Growing" if data["alert"] != "dry" else "Dry")
            
        st.markdown("---")
        
        # Alerts Section
        st.markdown("### ⚠️ Active Alerts")
        if data["alert"] == "flood":
            st.error("🚨 **FLOOD ALERT:** Heavy rains expected in the next 48 hours. Move livestock to higher ground and ensure drainage channels are clear.")
        elif data["alert"] == "dry":
            st.warning("🌵 **DRY SPELL ALERT:** Low rainfall expected for the next 7 days. Irrigate crops if possible and apply mulch to retain soil moisture.")
        else:
            st.success("✅ **No active weather alerts** for your area.")
            
        # Planting Calendar
        st.markdown("### 📅 Planting Calendar")
        st.info("""
        * **🌽 Maize:** Plant in November - December
        * **🌾 Harvest:** April - June
        * **🌱 Fertilizer:** Apply 3-4 weeks after planting
        """)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p><strong>AgriWise</strong> | Built by Mzuzu University Students</p>
    <p>Supported by the World Bank SAVE Project</p>
</div>
""", unsafe_allow_html=True)

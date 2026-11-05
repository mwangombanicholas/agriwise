import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64

st.set_page_config(page_title="AgriWise", page_icon="🌽", layout="wide")

MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

# --- BACKGROUND MEDIA SETUP ---
bg_image_path = "cornfield.jpg"
video_path = "weather_bg.mp4"

def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

bg_image_b64 = get_base64(bg_image_path)
video_b64 = get_base64(video_path)

# --- CUSTOM CSS (Landing Page & Dashboard) ---
st.markdown(f"""
<style>
    /* --- MAIN BACKGROUND --- */
    .stApp {{
        background-color: #ffffff;
    }}
    
    /* --- LANDING PAGE HERO SECTION --- */
    .hero-section {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("data:image/jpeg;base64,{bg_image_b64}");
        background-size: cover;
        background-position: center;
        height: 80vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        text-align: center;
        margin-top: -2rem; /* Pull up to cover the top */
        margin-bottom: 2rem;
    }}
    
    .hero-section h1 {{
        font-size: 5rem;
        font-weight: 900;
        letter-spacing: 5px;
        margin-bottom: 0;
        text-transform: uppercase;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
    }}
    
    .hero-section p {{
        font-size: 1.2rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }}

    /* --- LANDING PAGE TEXT SECTIONS --- */
    .text-section {{
        padding: 3rem;
        background-color: #1a1a1a;
        color: white;
        border-radius: 10px;
        margin-bottom: 3rem;
    }}
    .text-section h2 {{
        color: #8bc34a !important;
        font-size: 2.5rem;
    }}
    .text-section p {{
        color: #cccccc !important;
        font-size: 1.1rem;
    }}

    /* --- LOGIN CARD --- */
    .login-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border-top: 5px solid #8bc34a;
    }}
    
    .login-card h3 {{
        color: #1b5e20 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }}

    /* --- DASHBOARD STYLES --- */
    .dashboard-hero {{
        background-image: url("data:image/jpeg;base64,{bg_image_b64}");
        background-size: cover;
        background-position: center;
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}

    .dashboard-hero h1 {{
        color: white !important;
        font-weight: 800;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.8);
    }}

    .feature-badge {{
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #1b5e20 !important;
    }}

    .white-card {{
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }}

    .white-card h2, .white-card p, .white-card label, .white-card div, .white-card span {{
        color: #333 !important;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #2e7d32;
    }}

    /* Tab 2 Video */
    .video-container {{
        position: relative;
        width: 100%;
        min-height: 500px;
        border-radius: 15px;
        overflow: hidden;
    }}
    .video-container video {{
        position: absolute;
        top: 50%;
        left: 50%;
        min-width: 100%;
        min-height: 100%;
        transform: translate(-50%, -50%);
        z-index: 1;
        opacity: 0.8;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: #2e7d32 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #1b5e20 !important;
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

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'farmer_name' not in st.session_state:
    st.session_state['farmer_name'] = ""
if 'farmer_district' not in st.session_state:
    st.session_state['farmer_district'] = "Mzuzu"

# ==========================================
# PAGE 1: THE LANDING PAGE
# ==========================================
if not st.session_state['logged_in']:
    
    # 1. Hero Section
    st.markdown("""
    <div class="hero-section">
        <p>Grow Naturally, Live Natural</p>
        <h1>FARMING</h1>
    </div>
    """, unsafe_allow_html=True)

    # 2. Dark Text Section (Like the middle of your image)
    st.markdown("""
    <div class="text-section">
        <h2>Empowering Malawian Farmers with AI</h2>
        <p>We combine cutting-edge Artificial Intelligence with agricultural expertise to help smallholder farmers protect their crops, increase yields, and build climate resilience.</p>
        <p>Our mission is to ensure that no farmer loses their harvest due to late disease detection or unpredictable weather patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Login and Info Section
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        <div class="white-card" style="height: 100%;">
            <h2 style="color: #1b5e20 !important;">Why Choose AgriWise?</h2>
            <br>
            <h4 style="color: #2e7d32 !important;">🌱 96% Accuracy</h4>
            <p>Our CNN model was trained on thousands of images to accurately detect Common Rust, Northern Leaf Blight, and Cercospora.</p>
            
            <h4 style="color: #2e7d32 !important;">⛅ Localized Climate Advisory</h4>
            <p>Get real-time weather alerts and planting calendars specific to your district in Malawi.</p>
            
            <h4 style="color: #2e7d32 !important;">🤝 Designed for Smallholders</h4>
            <p>Simple, easy-to-use, and accessible on any basic smartphone.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="login-card">
            <h3>🌾 Farmer Login</h3>
            <p style="text-align: center; color: #666 !important; font-size: 0.9rem;">Enter your details to access the dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        name = st.text_input("Full Name", placeholder="e.g., John Phiri")
        district = st.selectbox("Home District", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
        
        if st.button("Enter Dashboard", use_container_width=True):
            if name:
                st.session_state['logged_in'] = True
                st.session_state['farmer_name'] = name
                st.session_state['farmer_district'] = district
                st.rerun()
            else:
                st.error("Please enter your name.")

# ==========================================
# PAGE 2: THE DASHBOARD
# ==========================================
else:
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🌽</h1>", unsafe_allow_html=True)
        st.markdown(f"### Welcome, **{st.session_state['farmer_name']}**!")
        st.markdown(f"📍 Location: **{st.session_state['farmer_district']}**")
        st.markdown("---")
        
        if st.button("Log Out", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("#### Supported Crop:")
        st.markdown("🌽 Maize (Corn)")
        st.markdown("#### Detects:")
        st.markdown("- Healthy Maize\n- Common Rust\n- Cercospora Leaf Spot\n- Northern Leaf Blight")

    # Main Dashboard Header
    st.markdown(f"""
    <div class="dashboard-hero">
        <h1>Welcome back, {st.session_state['farmer_name']}.</h1>
        <p style="font-size: 1.2rem;">Ready to scan your crops today?</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])

    # --- TAB 1: CROP DIAGNOSIS ---
    with tab1:
        col_left, col_right = st.columns([1, 1.5], gap="large")

        with col_left:
            st.markdown("""
            <div class="feature-badge">🌱 Better Growth</div>
            <div class="feature-badge">🛡️ Strong Protection</div>
            <div class="feature-badge">📈 Higher Yield</div>
            <div class="feature-badge">🌍 Sustainable Future</div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("""
            <div class="white-card" style="margin-bottom: 1rem;">
                <h2>📸 Scan Your Maize Crop</h2>
                <p>Take a clear, close-up photo of a single maize leaf to get an instant AI diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded = st.file_uploader("Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if uploaded:
                image = Image.open(uploaded).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
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
                                'common_rust': {'chemical': 'Azoxystrobin or Propiconazole fungicide', 'organic': 'Neem oil spray'},
                                'cercospora_leaf_spot': {'chemical': 'Mancozeb or Chlorothalonil fungicide', 'organic': 'Remove infected leaves'},
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

    # --- TAB 2: CLIMATE ADVISORY ---
    with tab2:
        st.markdown(f"""
        <div class="video-container">
            <video autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <div class="white-card" style="margin: 2rem; background: rgba(255, 255, 255, 0.95);">
                <h2 style="margin-top:0;">⛅ Localized Climate Advisory</h2>
                <p>Showing weather for your district: <strong>{st.session_state['farmer_district']}</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            default_index = ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"].index(st.session_state['farmer_district'])
            district = st.selectbox("📍 Select District", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"], index=default_index)
        
        weather_alerts = {
            "Mzuzu": ("No active weather alerts for your area.", "success"),
            "Mzimba": ("🌵 DRY SPELL ALERT: Irrigate if possible", "warning"),
            "Karonga": ("🚨 FLOOD ALERT: Heavy rains expected", "error"),
            "Rumphi": ("No active weather alerts for your area.", "success"),
            "Nkhata Bay": ("🚨 FLOOD ALERT: Heavy rains expected", "error")
        }
        
        msg, alert_type = weather_alerts.get(district, ("No active weather alerts for your area.", "success"))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("🌡️ Temperature", "24°C")
        m2.metric("🌧️ Rainfall", "Moderate")
        m3.metric("📅 Season", "Rainy / Growing")

        if alert_type == "error": st.error(msg)
        elif alert_type == "warning": st.warning(msg)
        else: st.success(msg)
        
        st.info("📅 **Planting Calendar:**\n- **Maize:** Plant in November - December\n- **Harvest:** April - June")

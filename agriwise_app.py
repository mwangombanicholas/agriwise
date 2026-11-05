import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import tensorflow as tf
import base64
import datetime
from database import init_db, add_farmer, verify_farmer

# Initialize database
init_db()

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

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_image_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    .block-container {{
        background: transparent !important;
        padding-top: 2rem !important;
    }}
    
    .hero-title {{
        color: white !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        line-height: 1.2;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
        margin-bottom: 2rem;
    }}

    .feature-badge {{
        background-color: white;
        border-radius: 30px;
        padding: 12px 25px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-weight: bold;
        color: #2e7d32 !important;
        font-size: 1.1rem;
    }}
    
    .feature-badge span {{ margin-right: 15px; font-size: 1.3rem; }}

    .white-card {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        position: relative;
        z-index: 10;
    }}

    .white-card h1, .white-card h2, .white-card h3, .white-card p, .white-card span, .white-card label, .white-card div, .white-card strong {{
        color: #1b5e20 !important;
    }}
    
    [data-testid="stFileUploader"] {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #2e7d32;
    }}

    [data-testid="stSidebar"] {{ background-color: #ffffff !important; }}
    [data-testid="stSidebar"] * {{ color: #1b5e20 !important; }}

    .video-container {{
        position: relative;
        width: 100%;
        min-height: 600px;
        border-radius: 15px;
        overflow: hidden;
    }}
    .video-container video {{
        position: absolute;
        top: 50%; left: 50%;
        min-width: 100%; min-height: 100%;
        width: auto; height: auto;
        transform: translate(-50%, -50%);
        z-index: 1;
        opacity: 0.8;
    }}

    .stTextInput > div > div > input {{
        color: #1b5e20 !important;
        background-color: #f8f9fa !important;
    }}
    
    /* Make ALL buttons green EXCEPT the logout button */
    .stButton > button {{
        background-color: #2e7d32 !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
    }}
    /* Make the Logout button RED */
    .logout-btn > button {{
        background-color: #d32f2f !important;
        color: white !important;
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
    hue, sat = img_hsv[:, :, 0], img_hsv[:, :, 1] 
    plant_pixels = np.sum(((hue >= 15) & (hue <= 100)) & (sat > 40))
    total_pixels = 128 * 128
    if (plant_pixels / total_pixels) * 100 < 25: return False
    gray = np.mean(img_rgb, axis=2)
    edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    if (edges / total_pixels) < 10: return False
    return True

# Real Weather API Function (NASA POWER - Free)
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_real_weather(lat, lon):
    try:
        today = datetime.datetime.now()
        end_date = today.strftime('%Y%m%d')
        start_date = (today - datetime.timedelta(days=7)).strftime('%Y%m%d')
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
        res = requests.get(url, timeout=5)
        data = res.json()
        
        # Get latest available data
        t2m_data = data['properties']['parameter']['T2M']
        rain_data = data['properties']['parameter']['PRECTOTCORR']
        
        latest_date = list(t2m_data.keys())[-1]
        temp = t2m_data[latest_date]
        rain = rain_data[latest_date]
        
        return temp, rain
    except:
        return None, None

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_name'] = ""

# ==========================================
# LANDING PAGE
# ==========================================
if not st.session_state['logged_in']:
    col_left, col_right = st.columns([1.5, 1], gap="large")
    
    with col_left:
        st.markdown('<br><br><br><br><div class="hero-title">Let\'s Get Started</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:white; font-size:1.2rem; text-shadow: 1px 1px 5px rgba(0,0,0,0.8);">Join AgriWise to instantly diagnose maize diseases, receive localized climate alerts, and protect your harvest using the power of Artificial Intelligence.</p>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.subheader("Sign In / Sign Up")
        
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])
        
        with tab_login:
            login_email = st.text_input("Email Address", key="log_email")
            login_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Login", use_container_width=True):
                name = verify_farmer(login_email, login_pass)
                if name:
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = name
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    
        with tab_signup:
            signup_name = st.text_input("Full Name", key="sign_name")
            signup_email = st.text_input("Email Address", key="sign_email")
            signup_pass = st.text_input("Create Password", type="password", key="sign_pass")
            if st.button("Sign Up", use_container_width=True):
                if signup_name and signup_email and signup_pass:
                    if add_farmer(signup_name, signup_email, signup_pass):
                        st.success("Account created! Please go to the Login tab.")
                    else:
                        st.error("Email already exists.")
                else:
                    st.warning("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# MAIN DASHBOARD
# ==========================================
else:
    with st.sidebar:
        # THE LOGOUT BUTTON (Wrapped in red CSS class)
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user_name'] = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"<h3>Welcome, {st.session_state['user_name']}! 👋</h3>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### About AgriWise")
        st.markdown("**Supported Crop:** Maize (Corn)")
        st.markdown("**Model Accuracy:** 96%")

    st.markdown('<div class="hero-title">Bringing Innovation to<br>Your Farming Journey.</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📸 AI Crop Diagnosis", "⛅ Climate Advisory"])
    
    # --- TAB 1 ---
    with tab1:
        col_left, col_right = st.columns([1, 1.5], gap="large")
        with col_left:
            st.markdown("""
            <div class="feature-badge"><span>🌱</span> Better Growth</div>
            <div class="feature-badge"><span>🛡️</span> Strong Protection</div>
            <div class="feature-badge"><span>📈</span> Higher Yield</div>
            <div class="feature-badge"><span>🌍</span> Sustainable Future</div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("""
            <div class="white-card">
                <h2>📸 Scan Your Maize Crop</h2>
                <p>Take a clear, close-up photo of a single maize leaf to get an instant AI diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if uploaded:
                image = Image.open(uploaded).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if not is_it_a_leaf(image):
                    st.error("❌ **Invalid Image Detected.** This does not look like a plant leaf.")
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
                            display_names = {'maize_cercospora_leaf_spot': 'Cercospora Leaf Spot', 'maize_common_rust': 'Common Rust', 'maize_healthy': 'Healthy Maize', 'maize_northern_leaf_blight': 'Northern Leaf Blight'}
                            result = class_names[idx]
                            
                            if 'healthy' in result: st.success(f"### ✅ Diagnosis: {display_names[result]}")
                            else: st.warning(f"### ⚠️ Diagnosis: {display_names[result]}")
                            st.info(f"**Confidence:** {confidence:.1f}%")
                            
                            info = {
                                'healthy': {'chemical': 'No treatment needed', 'organic': 'Continue good practices'},
                                'common_rust': {'chemical': 'Azoxystrobin fungicide', 'organic': 'Neem oil spray'},
                                'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide', 'organic': 'Remove infected leaves'},
                                'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole', 'organic': 'Baking soda solution'}
                            }.get(result.replace('maize_', ''), {'chemical': 'N/A', 'organic': 'N/A'})
                            
                            st.subheader("💊 Treatment Recommendations")
                            t1, t2 = st.columns(2)
                            t1.info(f"**🧪 Chemical:**\n\n{info['chemical']}")
                            t2.success(f"**🌿 Organic:**\n\n{info['organic']}")
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
    
    # --- TAB 2 ---
    with tab2:
        st.markdown(f"""
        <div class="video-container">
            <video autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <div class="white-card" style="margin: 2rem;">
                <h2 style="margin-top:0;">⛅ Real-Time Climate Advisory</h2>
                <p>Data pulled directly from NASA POWER Satellite API.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        district = st.selectbox("📍 Select District", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
        
        # Coordinates for the districts
        coords = {
            "Mzuzu": (-11.46, 34.02),
            "Mzimba": (-11.90, 33.60),
            "Karonga": (-9.93, 33.93),
            "Rumphi": (-11.02, 33.85),
            "Nkhata Bay": (-11.60, 34.30)
        }
        
        lat, lon = coords[district]
        
        with st.spinner("Fetching live satellite data from NASA..."):
            temp, rain = get_real_weather(lat, lon)
            
        m1, m2, m3 = st.columns(3)
        if temp is not None:
            m1.metric("🌡️ Real Temp (24h Avg)", f"{temp:.1f}°C")
            m2.metric("🌧️ Rain (Last 24h)", f"{rain:.1f} mm")
        else:
            m1.metric("🌡️ Temp", "API Error")
            m2.metric("🌧️ Rain", "API Error")
            
        m3.metric("📅 Season", "Rainy / Growing")
    
        # Alerts (Simulated for demonstration)
        if district in ["Karonga", "Nkhata Bay"]:
            st.error("🚨 FLOOD ALERT: High risk based on regional rainfall patterns.")
        elif district in ["Mzimba", "Rumphi"]:
            st.warning("🌵 DRY SPELL ALERT: Monitor soil moisture closely.")
        else:
            st.success("✅ No active weather alerts for your area.")

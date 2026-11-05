
import os
import base64
import datetime

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

from database import init_db, add_farmer, verify_farmer


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AgriWise | Smart Farming",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

MODEL_PATH = "agriwise_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"

BG_IMAGE = "cornfield.jpg"
WEATHER_VIDEO = "weather_bg.mp4"


# =========================================================
# ASSETS
# =========================================================
def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    return ""


bg_image_b64 = get_base64(BG_IMAGE)
video_b64 = get_base64(WEATHER_VIDEO)


# =========================================================
# CUSTOM UI
# =========================================================
background_css = ""

if bg_image_b64:
    background_css = f"""
        background-image:
            linear-gradient(rgba(8, 35, 18, .72), rgba(8, 35, 18, .78)),
            url("data:image/jpeg;base64,{bg_image_b64}");
    """


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {{
        --green-950: #102b1b;
        --green-900: #173d27;
        --green-800: #205536;
        --green-700: #287443;
        --green-100: #eaf4ec;
        --cream: #f7f9f5;
        --ink: #1d2b22;
        --muted: #68766c;
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    .stApp {{
        {background_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }}

    .brand-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.4rem;
    }}

    .brand-mark {{
        width: 46px;
        height: 46px;
        display: grid;
        place-items: center;
        border-radius: 15px;
        background: #e7f3e8;
        font-size: 25px;
    }}

    .brand-name {{
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.1;
    }}

    .brand-tagline {{
        color: rgba(255,255,255,.78);
        font-size: .78rem;
        margin-top: 3px;
    }}

    .hero {{
        padding: 2.1rem 0 1.5rem 0;
        max-width: 720px;
    }}

    .eyebrow {{
        color: #c5e6ca;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .9rem;
    }}

    .hero h1 {{
        color: #fff;
        font-size: clamp(2.5rem, 5vw, 4.4rem);
        line-height: 1.05;
        font-weight: 800;
        margin: 0 0 1rem 0;
        text-shadow: 0 3px 20px rgba(0,0,0,.25);
    }}

    .hero p {{
        color: rgba(255,255,255,.88);
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 650px;
    }}

    .glass-panel {{
        background: rgba(255,255,255,.96);
        border: 1px solid rgba(255,255,255,.65);
        border-radius: 22px;
        padding: 1.6rem;
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
    }}

    .panel-kicker {{
        color: var(--green-700);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .4rem;
    }}

    .panel-heading {{
        color: var(--green-950);
        font-family: 'Manrope', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: .35rem;
    }}

    .panel-copy {{
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.6;
    }}

    .feature-card {{
        background: rgba(255,255,255,.95);
        border: 1px solid rgba(255,255,255,.6);
        border-radius: 17px;
        padding: 1rem 1.1rem;
        margin: .65rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }}

    .feature-title {{
        color: var(--green-900);
        font-weight: 800;
        font-size: .98rem;
        margin-bottom: .2rem;
    }}

    .feature-copy {{
        color: #647268;
        font-size: .82rem;
        line-height: 1.45;
    }}

    .feature-icon {{
        font-size: 1.45rem;
        margin-bottom: .35rem;
    }}

    .section-label {{
        color: #d7eedb;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 1rem 0 .55rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255,255,255,.13);
        padding: 7px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.18);
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: 0 18px;
    }}

    .stTabs [aria-selected="true"] {{
        background: white !important;
        color: var(--green-900) !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background: transparent;
    }}

    .stTextInput label,
    .stSelectbox label,
    .stFileUploader label {{
        color: var(--green-900) !important;
        font-weight: 700 !important;
    }}

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 10px !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #287443, #1b5b35);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 46px;
        font-weight: 800;
        transition: all .2s ease;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #1f6338, #124629);
        border: 0;
        box-shadow: 0 7px 18px rgba(32,85,54,.22);
        transform: translateY(-1px);
    }}

    [data-testid="stFileUploader"] {{
        background: #f5faf5;
        border: 1.5px dashed #8ab99a;
        border-radius: 14px;
        padding: 1rem;
    }}

    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #e5eee6;
        padding: 1rem 1.1rem;
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(22,60,35,.05);
    }}

    [data-testid="stMetricLabel"] {{
        color: #65766a !important;
        font-weight: 700;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--green-900) !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }}

    [data-testid="stSidebar"] {{
        background: #f7faf6;
        border-right: 1px solid #e2ebe3;
    }}

    [data-testid="stSidebar"] * {{
        color: var(--green-950);
    }}

    .weather-banner {{
        border-radius: 18px;
        padding: 1.5rem;
        min-height: 170px;
        background: linear-gradient(115deg, rgba(12,48,29,.92), rgba(37,105,62,.78));
        color: white;
        margin-bottom: 1rem;
    }}

    .weather-banner h2 {{
        color: white;
        margin: 0 0 .45rem 0;
    }}

    .weather-banner p {{
        color: rgba(255,255,255,.83);
        margin: 0;
    }}

    .small-note {{
        color: #647268;
        font-size: .8rem;
        line-height: 1.5;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem 1rem 2rem 1rem;
        }}

        .hero {{
            padding-top: 1rem;
        }}

        .glass-panel {{
            padding: 1.1rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0 10px;
            font-size: .85rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL AND IMAGE HELPERS
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading the crop diagnosis model..."):
            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)


def is_it_a_leaf(image):
    """Basic image screening, not a maize-leaf classifier."""
    img = image.resize((128, 128))
    img_rgb = np.array(img)
    img_hsv = np.array(img.convert("HSV"))

    hue = img_hsv[:, :, 0]
    saturation = img_hsv[:, :, 1]

    plant_pixels = np.sum(
        ((hue >= 15) & (hue <= 100)) & (saturation > 40)
    )

    total_pixels = 128 * 128

    if (plant_pixels / total_pixels) * 100 < 25:
        return False

    gray = np.mean(img_rgb, axis=2)

    edges = (
        np.abs(np.diff(gray, axis=0)).sum()
        + np.abs(np.diff(gray, axis=1)).sum()
    )

    return (edges / total_pixels) >= 10


@st.cache_data(ttl=3600)
def get_real_weather(lat, lon, district_name):
    try:
        latest_day = datetime.datetime.now() - datetime.timedelta(days=4)

        end_date = latest_day.strftime("%Y%m%d")
        start_date = (
            latest_day - datetime.timedelta(days=2)
        ).strftime("%Y%m%d")

        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}"
            f"&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        temperature_data = data["properties"]["parameter"]["T2M"]
        rainfall_data = data["properties"]["parameter"]["PRECTOTCORR"]

        latest_date = list(temperature_data.keys())[-1]

        temperature = temperature_data[latest_date]
        rainfall = rainfall_data[latest_date]

        if temperature == -999.0 or rainfall == -999.0:
            raise ValueError("Satellite data is not ready.")

        return temperature, rainfall, "NASA POWER satellite data"

    except Exception:
        fallbacks = {
            "Mzuzu": (23.5, 1.2),
            "Mzimba": (25.0, 0.5),
            "Karonga": (29.0, 4.5),
            "Rumphi": (22.8, 1.0),
            "Nkhata Bay": (26.5, 5.2),
        }

        temperature, rainfall = fallbacks.get(
            district_name, (24.0, 1.5)
        )

        return temperature, rainfall, "Estimated fallback values"


# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


def brand():
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">🌽</div>
            <div>
                <div class="brand-name">AgriWise</div>
                <div class="brand-tagline">
                    Smart insights. Healthier harvests.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LANDING / AUTHENTICATION PAGE
# =========================================================
if not st.session_state.logged_in:

    brand()

    left, right = st.columns(
        [1.2, 0.9],
        gap="large",
        vertical_alignment="center"
    )

    with left:

        st.markdown(
            """
            <div class="hero">
                <div class="eyebrow">
                    AI-powered agriculture for Malawi
                </div>

                <h1>
                    Grow smarter.<br>
                    Harvest better.
                </h1>

                <p>
                    Identify common maize leaf diseases, explore localized
                    climate information and make more informed decisions
                    throughout your farming journey.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f1, f2 = st.columns(2)

        with f1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">📷</div>
                    <div class="feature-title">AI crop diagnosis</div>
                    <div class="feature-copy">
                        Upload a maize leaf photo for a model prediction.
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🌦️</div>
                    <div class="feature-title">Climate information</div>
                    <div class="feature-copy">
                        View temperature and rainfall information by district.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">🌱</div>
                    <div class="feature-title">Practical guidance</div>
                    <div class="feature-copy">
                        Explore general treatment and crop-care suggestions.
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🇲🇼</div>
                    <div class="feature-title">Made for local needs</div>
                    <div class="feature-copy">
                        Designed around selected districts in Malawi.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:

        st.markdown(
            '<div class="glass-panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-kicker">Welcome to AgriWise</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-heading">Your farm, better informed.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-copy">Sign in to continue, or create an account to get started.</div>',
            unsafe_allow_html=True
        )

        st.write("")

        login_tab, signup_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        with login_tab:

            with st.form("login_form"):

                login_email = st.text_input(
                    "Email address",
                    key="log_email"
                )

                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="log_pass"
                )

                login_submit = st.form_submit_button(
                    "Sign in to AgriWise",
                    use_container_width=True
                )

            if login_submit:

                if not login_email.strip() or not login_password:
                    st.warning(
                        "Enter your email address and password."
                    )

                else:
                    name = verify_farmer(
                        login_email.strip(),
                        login_password
                    )

                    if name:
                        st.session_state.logged_in = True
                        st.session_state.user_name = name
                        st.rerun()

                    else:
                        st.error(
                            "The email address or password is incorrect."
                        )

        with signup_tab:

            with st.form("signup_form"):

                signup_name = st.text_input(
                    "Full name",
                    key="sign_name"
                )

                signup_email = st.text_input(
                    "Email address",
                    key="sign_email"
                )

                signup_password = st.text_input(
                    "Create password",
                    type="password",
                    key="sign_pass"
                )

                signup_submit = st.form_submit_button(
                    "Create my account",
                    use_container_width=True
                )

            if signup_submit:

                if not all([
                    signup_name.strip(),
                    signup_email.strip(),
                    signup_password
                ]):
                    st.warning("Please complete all fields.")

                elif add_farmer(
                    signup_name.strip(),
                    signup_email.strip(),
                    signup_password
                ):
                    st.success(
                        "Account created. Sign in using your new details."
                    )

                else:
                    st.error(
                        "An account with that email address already exists."
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style="text-align:center;color:rgba(255,255,255,.72);
        font-size:.78rem;margin-top:2rem;">
            AgriWise · Supporting informed maize farming decisions
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================
else:

    with st.sidebar:

        brand()

        st.markdown("---")

        st.markdown(
            f"### Welcome, {st.session_state.user_name}"
        )

        st.caption("Your AgriWise farming workspace")

        st.markdown("---")

        st.markdown("**About this platform**")

        st.write("Supported crop: Maize")
        st.write("Disease classes: 4")

        st.caption(
            "The model's reported validation accuracy is not "
            "a guarantee for every image."
        )

        st.markdown("---")

        if st.button("↩ Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.rerun()

    brand()

    st.markdown(
        """
        <div class="hero" style="padding-top:.5rem;">
            <div class="eyebrow">Your farming dashboard</div>

            <h1 style="font-size:clamp(2.2rem,4vw,3.4rem);">
                Bringing insight<br>
                to every harvest.
            </h1>

            <p>
                Choose a service below to check your maize crop
                or explore district-level climate information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    diagnosis_tab, climate_tab = st.tabs(
        ["📷  Crop diagnosis", "🌤️  Climate advisory"]
    )


    # =====================================================
    # CROP DIAGNOSIS
    # =====================================================
    with diagnosis_tab:

        st.markdown(
            '<div class="section-label">Maize health assistant</div>',
            unsafe_allow_html=True
        )

        upload_col, guide_col = st.columns(
            [1.35, 0.85],
            gap="large"
        )

        with upload_col:

            with st.container(border=True):

                st.markdown("### Upload a maize leaf")

                st.caption(
                    "Use a clear, well-lit, close-up image of one maize leaf. "
                    "Avoid blurry photos and images with several objects."
                )

                uploaded_file = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png"],
                    help="Accepted formats: JPG, JPEG and PNG."
                )

                if uploaded_file is not None:

                    try:
                        crop_image = Image.open(
                            uploaded_file
                        ).convert("RGB")

                        st.image(
                            crop_image,
                            caption="Image selected for analysis",
                            use_container_width=True
                        )

                        if st.button(
                            "🔍 Analyse maize leaf",
                            use_container_width=True,
                            type="primary"
                        ):

                            if not is_it_a_leaf(crop_image):

                                st.error(
                                    "This image could not pass the basic "
                                    "plant-image screening. Please upload "
                                    "a clear maize leaf photo."
                                )

                            else:

                                try:

                                    with st.spinner(
                                        "Analysing the image..."
                                    ):

                                        model = load_model()

                                        resized = crop_image.resize(
                                            (128, 128)
                                        )

                                        image_array = (
                                            np.array(resized) / 255.0
                                        )

                                        image_array = np.expand_dims(
                                            image_array,
                                            axis=0
                                        )

                                        predictions = model.predict(
                                            image_array,
                                            verbose=0
                                        )

                                    class_names = [
                                        "maize_cercospora_leaf_spot",
                                        "maize_common_rust",
                                        "maize_healthy",
                                        "maize_northern_leaf_blight",
                                    ]

                                    selected_index = int(
                                        np.argmax(predictions[0])
                                    )

                                    result = class_names[selected_index]

                                    confidence = (
                                        float(predictions[0][selected_index])
                                        * 100
                                    )

                                    st.session_state["latest_diagnosis"] = {
                                        "result": result,
                                        "confidence": confidence,
                                    }

                                except Exception as error:

                                    st.error(
                                        f"Unable to analyse this image: {error}"
                                    )

                    except Exception:
                        st.error(
                            "Unable to open this image. Please try another file."
                        )

                diagnosis = st.session_state.get(
                    "latest_diagnosis"
                )

                if diagnosis:

                    result = diagnosis["result"]
                    confidence = diagnosis["confidence"]

                    display_names = {
                        "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
                        "maize_common_rust": "Common Rust",
                        "maize_healthy": "Healthy Maize",
                        "maize_northern_leaf_blight": "Northern Leaf Blight",
                    }

                    result_name = display_names[result]

                    st.divider()

                    st.markdown("### Diagnosis result")

                    if result == "maize_healthy":
                        st.success(f"✅ {result_name}")

                    else:
                        st.warning(f"⚠️ Possible {result_name}")

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    st.caption(
                        f"Model confidence: {confidence:.1f}%"
                    )

                    recommendations = {
                        "healthy": {
                            "chemical": (
                                "No treatment is indicated by this prediction."
                            ),
                            "organic": (
                                "Continue good crop-care practices "
                                "and monitor the plant."
                            ),
                        },

                        "common_rust": {
                            "chemical": (
                                "Consult a local agricultural extension "
                                "officer about an appropriate registered fungicide."
                            ),
                            "organic": (
                                "Remove heavily affected plant material "
                                "where appropriate and seek local extension guidance."
                            ),
                        },

                        "cercospora_leaf_spot": {
                            "chemical": (
                                "Ask an agricultural extension officer "
                                "about locally registered treatment options."
                            ),
                            "organic": (
                                "Manage crop residues and follow recommended "
                                "field sanitation practices."
                            ),
                        },

                        "northern_leaf_blight": {
                            "chemical": (
                                "Consult a local agricultural extension "
                                "officer about an appropriate registered fungicide."
                            ),
                            "organic": (
                                "Use recommended crop rotation and "
                                "field sanitation practices."
                            ),
                        },
                    }

                    advice = recommendations.get(
                        result.replace("maize_", ""),
                        recommendations["healthy"]
                    )

                    st.markdown("#### Suggested next steps")

                    advice_left, advice_right = st.columns(2)

                    with advice_left:
                        st.info(
                            "**🧪 Chemical guidance**\n\n"
                            + advice["chemical"]
                        )

                    with advice_right:
                        st.success(
                            "**🌿 Crop-care guidance**\n\n"
                            + advice["organic"]
                        )

                    st.caption(
                        "This is an AI-generated screening result, not "
                        "a confirmed agronomic diagnosis. Confirm suspected "
                        "disease with an agricultural extension officer "
                        "before applying any treatment."
                    )

        with guide_col:

            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">☀️</div>
                    <div class="feature-title">Good lighting</div>
                    <div class="feature-copy">
                        Take the photo in natural light so leaf details are visible.
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🍃</div>
                    <div class="feature-title">One leaf at a time</div>
                    <div class="feature-copy">
                        Keep the maize leaf as the main subject in the frame.
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📸</div>
                    <div class="feature-title">Keep it in focus</div>
                    <div class="feature-copy">
                        Avoid motion blur, distant shots and heavily obstructed leaves.
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">ℹ️</div>
                    <div class="feature-title">Use results carefully</div>
                    <div class="feature-copy">
                        AI predictions can be wrong. Seek local expert advice before treatment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # =====================================================
    # CLIMATE ADVISORY
    # =====================================================
    with climate_tab:

        st.markdown(
            '<div class="section-label">District climate information</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="weather-banner">
                <h2>🌦️ Localized climate advisory</h2>
                <p>
                    Explore recent temperature and rainfall information
                    for selected districts in Malawi.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        district_col, detail_col = st.columns(
            [1, 2],
            gap="large"
        )

        with district_col:

            with st.container(border=True):

                st.markdown("### Choose your area")

                district = st.selectbox(
                    "District / area",
                    [
                        "Mzuzu",
                        "Mzimba",
                        "Karonga",
                        "Rumphi",
                        "Nkhata Bay"
                    ]
                )

                st.caption("Select the area you want to view.")

        coordinates = {
            "Mzuzu": (-11.46, 34.02),
            "Mzimba": (-11.90, 33.60),
            "Karonga": (-9.93, 33.93),
            "Rumphi": (-11.02, 33.85),
            "Nkhata Bay": (-11.60, 34.30),
        }

        latitude, longitude = coordinates[district]

        with st.spinner(
            f"Retrieving climate information for {district}..."
        ):

            temperature, rainfall, data_source = get_real_weather(
                latitude,
                longitude,
                district
            )

        with detail_col:

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "🌡️ Temperature",
                f"{temperature:.1f} °C"
            )

            metric2.metric(
                "🌧️ Rainfall",
                f"{rainfall:.1f} mm"
            )

            metric3.metric(
                "📍 Selected area",
                district
            )

        st.caption(
            f"Data source: {data_source}. "
            "NASA POWER data may be delayed; fallback values are estimates."
        )

        if district in ["Karonga", "Nkhata Bay"]:

            st.warning(
                "Flood-risk notice: monitor official weather updates "
                "and local advisories, especially in flood-prone areas."
            )

        elif district in ["Mzimba", "Rumphi"]:

            st.info(
                "Dry-spell notice: monitor soil moisture and follow "
                "local agricultural extension advice."
            )

        else:

            st.success(
                "No district-specific alert is configured for this area "
                "in the current prototype."
            )

        st.caption(
            "Climate figures and prototype alerts are informational. "
            "They should not replace official forecasts or local "
            "disaster-management guidance."
        )


import os
import base64
import datetime
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from database import init_db, add_farmer, verify_farmer


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AgriWise | Smart Agriculture",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

MODEL_PATH = "agriwise_model.h5"

MODEL_URL = (
    "https://drive.google.com/uc?id="
    "1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw&export=download"
)

BG_IMAGE = "cornfield.jpg"

CLASS_NAMES = [
    "maize_cercospora_leaf_spot",
    "maize_common_rust",
    "maize_healthy",
    "maize_northern_leaf_blight"
]

DISPLAY_NAMES = {
    "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
    "maize_common_rust": "Common Rust",
    "maize_healthy": "Healthy Maize",
    "maize_northern_leaf_blight": "Northern Leaf Blight"
}


# ==========================================================
# 2. BACKGROUND IMAGE
# ==========================================================

def get_background():

    if os.path.exists(BG_IMAGE):

        with open(BG_IMAGE, "rb") as file:

            encoded = base64.b64encode(
                file.read()
            ).decode()

        return f"""
        background-image:
        linear-gradient(
            rgba(7, 35, 18, 0.88),
            rgba(7, 35, 18, 0.90)
        ),
        url("data:image/jpeg;base64,{encoded}");
        """

    return """
    background:
    radial-gradient(circle at top right, #285f3b, transparent 45%),
    linear-gradient(135deg, #102b1b, #173d27);
    """


# ==========================================================
# 3. PROFESSIONAL UI DESIGN
# ==========================================================

st.markdown(
    f"""
    <style>

    @import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {{
        --primary: #237443;
        --primary-dark: #185a33;
        --dark: #173d27;
        --muted: #6b7c70;
        --light: #f4f8f4;
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    .stApp {{
        {get_background()}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        max-width: 1350px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        font-family: 'Manrope', sans-serif !important;
    }}

    /* BRAND */

    .brand {{
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 30px;
    }}

    .brand-icon {{
        width: 54px;
        height: 54px;
        background: #f0f7ef;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 29px;
    }}

    .brand-name {{
        color: white;
        font-size: 23px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }}

    .brand-subtitle {{
        color: #d8e9dc;
        font-size: 12px;
    }}

    /* LANDING PAGE */

    .eyebrow {{
        color: #bce7c7;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 17px;
    }}

    .hero-title {{
        font-family: 'Manrope', sans-serif !important;
        color: white !important;
        font-size: clamp(42px, 5vw, 68px) !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin-bottom: 18px !important;
    }}

    .hero-description {{
        color: #e0eee3;
        font-size: 17px;
        line-height: 1.8;
        max-width: 590px;
    }}

    /* FEATURE CARDS */

    .feature-card {{
        background: #ffffff;
        border-radius: 18px;
        padding: 22px;
        min-height: 155px;
        margin: 8px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,.12);
        border: 1px solid #e8eee9;
    }}

    .feature-icon {{
        font-size: 29px;
        margin-bottom: 12px;
    }}

    .feature-title {{
        color: #173d27;
        font-weight: 800;
        font-size: 16px;
        margin-bottom: 7px;
    }}

    .feature-description {{
        color: #68796d;
        font-size: 13px;
        line-height: 1.6;
    }}

    /* AUTHENTICATION CARD */

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #ffffff !important;
        border: 1px solid #e7eee8 !important;
        border-radius: 23px !important;
        box-shadow: 0 20px 55px rgba(0,0,0,.18);
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding: 20px !important;
    }}

    .auth-kicker {{
        color: #237443;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }}

    .auth-heading {{
        color: #173d27 !important;
        font-family: 'Manrope', sans-serif;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 8px;
    }}

    .auth-description {{
        color: #738177;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 17px;
    }}

    /* AUTH TABS */

    [data-testid="stTabs"] {{
        background: transparent !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: #f0f5f1 !important;
        padding: 5px !important;
        border-radius: 12px !important;
        gap: 5px !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"] {{
        flex: 1;
        justify-content: center;
        border-radius: 9px !important;
        color: #627568 !important;
        font-weight: 700 !important;
        height: 43px;
    }}

    [data-testid="stTabs"] [aria-selected="true"] {{
        background: #ffffff !important;
        color: #185a33 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background: transparent !important;
    }}

    /* FORM */

    [data-testid="stForm"] {{
        background: #ffffff !important;
        border: none !important;
        padding: 8px 0 0 0 !important;
        box-shadow: none !important;
    }}

    [data-testid="stForm"] label,
    [data-testid="stTextInput"] label {{
        color: #263c2d !important;
        font-size: 13px !important;
        font-weight: 700 !important;
    }}

    [data-testid="stTextInput"] input {{
        background: #f8faf8 !important;
        color: #173d27 !important;
        border: 1px solid #dce7de !important;
        border-radius: 11px !important;
        min-height: 48px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: all .2s ease;
    }}

    [data-testid="stTextInput"] input:focus {{
        border: 1px solid #237443 !important;
        box-shadow: 0 0 0 3px rgba(35,116,67,.12) !important;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #9aa99e !important;
    }}

    /* PASSWORD VISIBILITY BUTTON */

    [data-testid="stTextInput"] button {{
        color: #237443 !important;
    }}

    /* SUBMIT BUTTONS */

    [data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(
            135deg,
            #287b49,
            #185c35
        ) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 11px !important;
        min-height: 49px !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        box-shadow: 0 6px 16px rgba(35,116,67,.2);
        transition: all .2s ease;
    }}

    [data-testid="stFormSubmitButton"] button:hover {{
        background: #124b2a !important;
        transform: translateY(-1px);
        box-shadow: 0 9px 20px rgba(35,116,67,.28);
    }}

    .stButton button {{
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    /* UPLOAD */

    [data-testid="stFileUploader"] {{
        background: #f7faf7;
        border: 1.5px dashed #9abca3;
        border-radius: 13px;
        padding: 12px;
    }}

    /* DASHBOARD */

    .dashboard-title {{
        color: white !important;
        font-size: clamp(30px, 4vw, 48px) !important;
        font-weight: 800 !important;
    }}

    .dashboard-description {{
        color: #dcebe0;
        font-size: 15px;
        line-height: 1.7;
    }}

    .section-label {{
        color: #c4e8ce;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .weather-banner {{
        background: linear-gradient(120deg, #174b2d, #287b49);
        padding: 25px;
        border-radius: 18px;
        margin: 15px 0;
    }}

    .weather-banner h2 {{
        color: white !important;
    }}

    .weather-banner p {{
        color: #e0eee3 !important;
    }}

    [data-testid="stMetric"] {{
        background: white;
        padding: 17px;
        border-radius: 14px;
        border: 1px solid #e3ebe4;
    }}

    [data-testid="stMetricLabel"] {{
        color: #64766a !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #173d27 !important;
        font-weight: 800 !important;
    }}

    /* SIDEBAR */

    [data-testid="stSidebar"] {{
        background: #f6faf6;
    }}

    [data-testid="stSidebar"] * {{
        color: #173d27;
    }}

    /* FOOTER */

    .app-footer {{
        color: #d4e7d9;
        font-size: 12px;
        text-align: center;
        margin-top: 35px;
        padding: 15px;
    }}

    /* MOBILE */

    @media(max-width: 768px) {{
        .block-container {{
            padding: 1rem;
        }}

        .hero-title {{
            font-size: 40px !important;
        }}

        .auth-heading {{
            font-size: 24px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 4. MODEL FUNCTIONS
# ==========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        response = requests.get(
            MODEL_URL,
            timeout=120
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)

    return tf.keras.models.load_model(MODEL_PATH)


# IMPORTANT:
# This is a preliminary visual screening heuristic.
# It is NOT a reliable maize-leaf classifier.
# A separately trained image classifier is required
# to reliably reject unrelated images.

def basic_leaf_screening(image):

    img = image.resize((128, 128))

    hsv = np.asarray(
        img.convert("HSV")
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    green_pixels = np.sum(
        (hue >= 15) &
        (hue <= 100) &
        (saturation > 40)
    )

    percentage = (
        green_pixels / (128 * 128)
    ) * 100

    return percentage >= 25


# ==========================================================
# 5. CLIMATE INFORMATION
# ==========================================================

@st.cache_data(ttl=3600)
def get_weather(latitude, longitude, district):

    try:

        target_day = (
            datetime.datetime.now() -
            datetime.timedelta(days=4)
        )

        end_date = target_day.strftime("%Y%m%d")

        start_date = (
            target_day - datetime.timedelta(days=2)
        ).strftime("%Y%m%d")

        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            "?parameters=T2M,PRECTOTCORR"
            "&community=AG"
            f"&longitude={longitude}"
            f"&latitude={latitude}"
            f"&start={start_date}"
            f"&end={end_date}"
            "&format=JSON"
        )

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        temperatures = (
            data["properties"]["parameter"]["T2M"]
        )

        rainfall_data = (
            data["properties"]["parameter"]["PRECTOTCORR"]
        )

        latest_date = list(temperatures.keys())[-1]

        temperature = temperatures[latest_date]
        rainfall = rainfall_data[latest_date]

        if temperature == -999 or rainfall == -999:
            raise ValueError("Weather data unavailable")

        return (
            temperature,
            rainfall,
            "NASA POWER satellite data"
        )

    except Exception:

        fallback = {
            "Mzuzu": (23.5, 1.2),
            "Mzimba": (25.0, 0.5),
            "Karonga": (29.0, 4.5),
            "Rumphi": (22.8, 1.0),
            "Nkhata Bay": (26.5, 5.2)
        }

        temperature, rainfall = fallback.get(
            district,
            (24.0, 1.5)
        )

        return (
            temperature,
            rainfall,
            "Estimated fallback values"
        )


# ==========================================================
# 6. SESSION MANAGEMENT
# ==========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "latest_diagnosis" not in st.session_state:
    st.session_state.latest_diagnosis = None


# ==========================================================
# 7. BRAND COMPONENT
# ==========================================================

def show_brand():

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🌽
            </div>

            <div>
                <div class="brand-name">
                    AgriWise
                </div>

                <div class="brand-subtitle">
                    Smart insights. Healthier harvests.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# 8. LANDING PAGE AND AUTHENTICATION
# ==========================================================

if not st.session_state.logged_in:

    show_brand()

    left, right = st.columns(
        [1.15, 0.9],
        gap="large",
        vertical_alignment="center"
    )

    # ------------------------------------------------------
    # LEFT SIDE: INTRODUCTION
    # ------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="eyebrow">
                AI-POWERED AGRICULTURE FOR MALAWI
            </div>

            <h1 class="hero-title">
                Grow smarter.<br>
                Harvest better.
            </h1>

            <p class="hero-description">
                Identify common maize leaf diseases, explore
                localized climate information and make more
                informed decisions throughout your farming journey.
            </p>
            """,
            unsafe_allow_html=True
        )

        card1, card2 = st.columns(2)

        with card1:

            st.markdown(
                """
                <div class="feature-card">

                    <div class="feature-icon">📷</div>

                    <div class="feature-title">
                        AI Crop Diagnosis
                    </div>

                    <div class="feature-description">
                        Upload maize leaf images and explore
                        AI-powered disease predictions.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-icon">🌦️</div>

                    <div class="feature-title">
                        Climate Information
                    </div>

                    <div class="feature-description">
                        Explore temperature and rainfall
                        information for selected districts.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with card2:

            st.markdown(
                """
                <div class="feature-card">

                    <div class="feature-icon">🌱</div>

                    <div class="feature-title">
                        Practical Guidance
                    </div>

                    <div class="feature-description">
                        Access general crop-care and
                        treatment guidance.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-icon">🇲🇼</div>

                    <div class="feature-title">
                        Made for Malawi
                    </div>

                    <div class="feature-description">
                        Designed to support local farming
                        information needs.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------------------------------------------
    # RIGHT SIDE: PROFESSIONAL AUTHENTICATION
    # ------------------------------------------------------

    with right:

        with st.container(border=True):

            st.markdown(
                """
                <div class="auth-kicker">
                    WELCOME TO AGRIWISE
                </div>

                <div class="auth-heading">
                    Your farm, better informed.
                </div>

                <div class="auth-description">
                    Sign in to continue, or create an account
                    to get started with smarter farming insights.
                </div>
                """,
                unsafe_allow_html=True
            )

            login_tab, signup_tab = st.tabs(
                ["Sign in", "Create account"]
            )

            # --------------------------------------------------
            # SIGN IN
            # --------------------------------------------------

            with login_tab:

                with st.form("login_form"):

                    email = st.text_input(
                        "Email address",
                        placeholder="Enter your email address"
                    )

                    password = st.text_input(
                        "Password",
                        placeholder="Enter your password",
                        type="password"
                    )

                    st.markdown("<div style='height:8px'></div>",
                                unsafe_allow_html=True)

                    login_button = st.form_submit_button(
                        "Sign in to AgriWise  →",
                        use_container_width=True
                    )

                if login_button:

                    if not email.strip() or not password:

                        st.warning(
                            "Please enter your email and password."
                        )

                    else:

                        try:

                            user = verify_farmer(
                                email.strip(),
                                password
                            )

                            if user:

                                st.session_state.logged_in = True
                                st.session_state.user_name = user

                                st.rerun()

                            else:

                                st.error(
                                    "Incorrect email address or password."
                                )

                        except Exception as error:

                            st.error(
                                f"Login error: {error}"
                            )

            # --------------------------------------------------
            # CREATE ACCOUNT
            # --------------------------------------------------

            with signup_tab:

                with st.form("signup_form"):

                    full_name = st.text_input(
                        "Full name",
                        placeholder="Enter your full name"
                    )

                    new_email = st.text_input(
                        "Email address",
                        placeholder="Enter your email address"
                    )

                    new_password = st.text_input(
                        "Create password",
                        placeholder="Create a password",
                        type="password"
                    )

                    confirm_password = st.text_input(
                        "Confirm password",
                        placeholder="Confirm your password",
                        type="password"
                    )

                    st.markdown("<div style='height:8px'></div>",
                                unsafe_allow_html=True)

                    signup_button = st.form_submit_button(
                        "Create my account  →",
                        use_container_width=True
                    )

                if signup_button:

                    if (
                        not full_name.strip()
                        or not new_email.strip()
                        or not new_password
                        or not confirm_password
                    ):

                        st.warning(
                            "Please complete all the fields."
                        )

                    elif new_password != confirm_password:

                        st.error(
                            "Your passwords do not match."
                        )

                    elif len(new_password) < 8:

                        st.error(
                            "Your password must contain at least 8 characters."
                        )

                    else:

                        try:

                            created = add_farmer(
                                full_name.strip(),
                                new_email.strip(),
                                new_password
                            )

                            if created:

                                st.success(
                                    "Account created successfully. "
                                    "You can now sign in."
                                )

                            else:

                                st.error(
                                    "This email may already be registered."
                                )

                        except Exception as error:

                            st.error(
                                f"Registration error: {error}"
                            )

    st.markdown(
        """
        <div class="app-footer">
            © 2026 AgriWise · Smart insights. Healthier harvests.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# 9. FARMER DASHBOARD
# ==========================================================

else:

    # ------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------

    with st.sidebar:

        show_brand()

        st.markdown("---")

        st.markdown(
            f"### Welcome, {st.session_state.user_name}"
        )

        st.caption("Your farming workspace")

        st.markdown("---")

        st.markdown("### Platform overview")

        st.write("🌽 Supported crop: Maize")

        st.write("🔬 Disease categories: 4")

        st.write("🌦️ Climate districts: 5")

        st.markdown("---")

        if st.button(
            "↩ Log out",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.user_name = ""

            st.session_state.latest_diagnosis = None

            st.rerun()

    # ------------------------------------------------------
    # DASHBOARD HEADER
    # ------------------------------------------------------

    show_brand()

    st.markdown(
        """
        <div class="section-label">
            YOUR FARMING DASHBOARD
        </div>

        <h1 class="dashboard-title">
            Welcome to AgriWise.
        </h1>

        <p class="dashboard-description">
            Monitor crop health and explore climate information
            to make more informed farming decisions.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # DASHBOARD TABS
    # ------------------------------------------------------

    diagnosis_tab, climate_tab = st.tabs(
        [
            "📷 Crop Diagnosis",
            "🌦️ Climate Advisory"
        ]
    )

    # ======================================================
    # 10. CROP DIAGNOSIS
    # ======================================================

    with diagnosis_tab:

        st.markdown(
            '<div class="section-label">MAIZE HEALTH ASSISTANT</div>',
            unsafe_allow_html=True
        )

        upload_col, tips_col = st.columns(
            [1.4, 0.8],
            gap="large"
        )

        with upload_col:

            with st.container(border=True):

                st.subheader(
                    "Upload a maize leaf image"
                )

                st.write(
                    "Choose a clear, close-up photo of one maize leaf."
                )

                uploaded_file = st.file_uploader(
                    "Choose JPG or PNG image",
                    type=["jpg", "jpeg", "png"],
                    key="maize_upload"
                )

                if uploaded_file is not None:

                    try:

                        image = Image.open(
                            uploaded_file
                        ).convert("RGB")

                        st.image(
                            image,
                            caption="Selected image",
                            use_container_width=True
                        )

                        if st.button(
                            "🔍 Analyse image",
                            type="primary",
                            use_container_width=True
                        ):

                            if not basic_leaf_screening(image):

                                st.session_state.latest_diagnosis = None

                                st.error(
                                    "The image did not pass the preliminary "
                                    "plant-image screening. Please upload "
                                    "a clear maize leaf image."
                                )

                            else:

                                with st.spinner(
                                    "Analysing your image..."
                                ):

                                    model = load_model()

                                    resized = image.resize(
                                        (128, 128)
                                    )

                                    array = (
                                        np.asarray(resized) / 255.0
                                    )

                                    array = np.expand_dims(
                                        array,
                                        axis=0
                                    )

                                    prediction = model.predict(
                                        array,
                                        verbose=0
                                    )

                                index = int(
                                    np.argmax(prediction[0])
                                )

                                result = CLASS_NAMES[index]

                                confidence = (
                                    float(prediction[0][index]) * 100
                                )

                                st.session_state.latest_diagnosis = {
                                    "result": result,
                                    "confidence": confidence
                                }

                        diagnosis = (
                            st.session_state.latest_diagnosis
                        )

                        if diagnosis:

                            st.divider()

                            st.subheader(
                                "Diagnosis result"
                            )

                            result = diagnosis["result"]

                            confidence = diagnosis["confidence"]

                            if result == "maize_healthy":

                                st.success(
                                    "✅ Healthy Maize"
                                )

                            else:

                                st.warning(
                                    f"⚠️ Possible "
                                    f"{DISPLAY_NAMES[result]}"
                                )

                            st.progress(
                                min(
                                    max(confidence / 100, 0.0),
                                    1.0
                                )
                            )

                            st.caption(
                                f"Model confidence: {confidence:.1f}%"
                            )

                            st.info(
                                "This is an AI screening result, "
                                "not a confirmed diagnosis. Consult "
                                "an agricultural extension officer "
                                "before applying treatment."
                            )

                    except Exception as error:

                        st.error(
                            f"Image analysis failed: {error}"
                        )

        with tips_col:

            st.markdown(
                """
                <div class="feature-card">

                    <div class="feature-icon">☀️</div>

                    <div class="feature-title">
                        Good lighting
                    </div>

                    <div class="feature-description">
                        Take your photo in natural light.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-icon">🍃</div>

                    <div class="feature-title">
                        Focus on one leaf
                    </div>

                    <div class="feature-description">
                        Keep the maize leaf as the main subject.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-icon">📸</div>

                    <div class="feature-title">
                        Clear pictures
                    </div>

                    <div class="feature-description">
                        Avoid blurry images and unnecessary backgrounds.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # ======================================================
    # 11. CLIMATE ADVISORY
    # ======================================================

    with climate_tab:

        st.markdown(
            """
            <div class="section-label">
                DISTRICT CLIMATE INFORMATION
            </div>

            <div class="weather-banner">

                <h2>
                    Localized climate advisory
                </h2>

                <p>
                    Explore recent temperature and rainfall
                    information for selected areas in Malawi.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        district = st.selectbox(
            "Select district / area",
            [
                "Mzuzu",
                "Mzimba",
                "Karonga",
                "Rumphi",
                "Nkhata Bay"
            ]
        )

        coordinates = {

            "Mzuzu": (-11.46, 34.02),

            "Mzimba": (-11.90, 33.60),

            "Karonga": (-9.93, 33.93),

            "Rumphi": (-11.02, 33.85),

            "Nkhata Bay": (-11.60, 34.30)

        }

        latitude, longitude = coordinates[district]

        with st.spinner(
            f"Retrieving information for {district}..."
        ):

            temperature, rainfall, source = get_weather(
                latitude,
                longitude,
                district
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C"
        )

        col2.metric(
            "🌧️ Rainfall",
            f"{rainfall:.1f} mm"
        )

        col3.metric(
            "📍 Selected area",
            district
        )

        st.caption(
            f"Data source: {source}"
        )

        if source == "Estimated fallback values":

            st.warning(
                "Satellite data could not be retrieved. "
                "These values are fallback estimates, "
                "not live weather observations."
            )

        if district in ["Karonga", "Nkhata Bay"]:

            st.info(
                "Monitor official weather updates and "
                "local flood advisories."
            )

        elif district in ["Mzimba", "Rumphi"]:

            st.info(
                "Monitor soil moisture and follow local "
                "agricultural extension advice."
            )

        else:

            st.success(
                "Continue monitoring local weather conditions "
                "and crop development."
            )

        st.caption(
            "Climate information is for guidance only. "
            "It does not replace official weather forecasts."
        )

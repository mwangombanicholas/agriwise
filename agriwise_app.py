
import os
import datetime
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from database import init_db, add_farmer, verify_farmer


# =========================================================
# 1. PAGE SETUP
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

CLASS_NAMES = [
    "maize_cercospora_leaf_spot",
    "maize_common_rust",
    "maize_healthy",
    "maize_northern_leaf_blight",
]

DISPLAY_NAMES = {
    "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
    "maize_common_rust": "Common Rust",
    "maize_healthy": "Healthy Maize",
    "maize_northern_leaf_blight": "Northern Leaf Blight",
}


# =========================================================
# 2. CUSTOM STYLING
# =========================================================
def load_background(path):
    if not os.path.exists(path):
        return ""
    import base64
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


background_image = load_background(BG_IMAGE)

if background_image:
    app_background = f"""
        background-image:
            linear-gradient(rgba(8, 35, 18, .82), rgba(8, 35, 18, .86)),
            url("data:image/jpeg;base64,{background_image}");
    """
else:
    app_background = """
        background:
            radial-gradient(circle at top right, #285f3b 0%, transparent 38%),
            linear-gradient(135deg, #102b1b, #173d27);
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {{
        --green-dark: #143923;
        --green: #287443;
        --green-light: #eaf4ec;
        --ink: #1c2b21;
        --muted: #66756b;
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    .stApp {{
        {app_background}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.025em;
    }}

    /* Brand */
    .brand {{
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 1.5rem;
    }}

    .brand-icon {{
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        background: #f0f7ef;
        font-size: 27px;
    }}

    .brand-title {{
        color: white;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.35rem;
        line-height: 1.15;
    }}

    .brand-subtitle {{
        color: rgba(255,255,255,.82);
        font-size: .82rem;
        margin-top: 3px;
    }}

    /* Landing hero */
    .eyebrow {{
        color: #d4edda;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: .8rem 0 1rem 0;
    }}

    .hero-title {{
        color: white !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: clamp(2.6rem, 5vw, 4.6rem) !important;
        line-height: 1.08 !important;
        font-weight: 800 !important;
        margin: 0 0 1rem 0 !important;
        text-shadow: 0 3px 18px rgba(0,0,0,.25);
    }}

    .hero-copy {{
        color: rgba(255,255,255,.9) !important;
        font-size: 1.05rem;
        line-height: 1.75;
        max-width: 650px;
    }}

    /* Panels and feature cards */
    div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,.97);
        border: 1px solid rgba(255,255,255,.65);
        border-radius: 18px;
    }}

    div[data-testid="stForm"] {{
        padding: 1.2rem;
    }}

    .feature-card {{
        background: rgba(255,255,255,.97);
        border: 1px solid rgba(255,255,255,.6);
        border-radius: 17px;
        padding: 1.15rem;
        margin: .55rem 0;
        min-height: 145px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }}

    .feature-icon {{
        font-size: 1.6rem;
        margin-bottom: .55rem;
    }}

    .feature-title {{
        color: #173d27;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: .35rem;
    }}

    .feature-copy {{
        color: #66756b;
        font-size: .86rem;
        line-height: 1.5;
    }}

    .panel-kicker {{
        color: #287443;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }}

    .panel-title {{
        color: #173d27;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        margin: .35rem 0;
    }}

    .panel-copy {{
        color: #66756b;
        font-size: .9rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }}

    /* Form labels and fields */
    [data-testid="stForm"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stFileUploader"] label {{
        color: #173d27 !important;
        font-weight: 700 !important;
    }}

    [data-testid="stTextInput"] input {{
        background: #f7faf7 !important;
        color: #172b1e !important;
        border: 1px solid #d8e5da !important;
        border-radius: 10px !important;
        min-height: 44px;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #849188 !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255,255,255,.14);
        padding: 7px;
        border-radius: 13px;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: white;
        font-weight: 700;
        border-radius: 9px;
        padding: 0 17px;
        min-height: 44px;
    }}

    .stTabs [aria-selected="true"] {{
        background: white !important;
        color: #173d27 !important;
    }}

    /* Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        background: #287443 !important;
        color: white !important;
        border: 1px solid #287443 !important;
        border-radius: 10px !important;
        min-height: 45px;
        font-weight: 800;
    }}

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        background: #1c5b34 !important;
        border-color: #1c5b34 !important;
    }}

    /* Upload and metrics */
    [data-testid="stFileUploader"] {{
        background: #f5faf5;
        border: 1.5px dashed #8ab99a;
        border-radius: 13px;
        padding: .8rem;
    }}

    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #e5eee6;
        padding: 1rem;
        border-radius: 14px;
    }}

    [data-testid="stMetricLabel"] {{
        color: #65766a !important;
        font-weight: 700;
    }}

    [data-testid="stMetricValue"] {{
        color: #173d27 !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }}

    /* Dashboard text */
    .dashboard-heading {{
        color: white !important;
        font-size: clamp(2.1rem, 4vw, 3.4rem) !important;
        line-height: 1.1 !important;
        font-weight: 800 !important;
    }}

    .dashboard-copy {{
        color: rgba(255,255,255,.88) !important;
        line-height: 1.6;
    }}

    .section-label {{
        color: #d4edda;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 1rem 0 .65rem;
    }}

    .weather-banner {{
        background: linear-gradient(115deg, #143923, #287443);
        border-radius: 17px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}

    .weather-banner h2 {{
        color: white !important;
        margin: 0 0 .4rem 0;
    }}

    .weather-banner p {{
        color: rgba(255,255,255,.86) !important;
        margin: 0;
    }}

    [data-testid="stSidebar"] {{
        background: #f7faf6;
    }}

    [data-testid="stSidebar"] * {{
        color: #173d27;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem 1rem 2rem 1rem;
        }}
        .hero-title {{
            font-size: 2.5rem !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MODEL AND WEATHER FUNCTIONS
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading the AI model..."):
            response = requests.get(MODEL_URL, timeout=90)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:
                file.write(response.content)

    return tf.keras.models.load_model(MODEL_PATH)


def basic_leaf_screening(image):
    """
    This is only a basic visual filter.
    It cannot confirm that an image is a maize leaf.
    """
    img = image.resize((128, 128))
    rgb = np.asarray(img)
    hsv = np.asarray(img.convert("HSV"))

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    green_or_yellow_pixels = np.sum(
        ((hue >= 15) & (hue <= 100)) & (saturation > 40)
    )

    if (green_or_yellow_pixels / (128 * 128)) * 100 < 25:
        return False

    gray = np.mean(rgb, axis=2)
    edges = (
        np.abs(np.diff(gray, axis=0)).sum()
        + np.abs(np.diff(gray, axis=1)).sum()
    )

    return (edges / (128 * 128)) >= 10


@st.cache_data(ttl=3600)
def get_weather(latitude, longitude, district):
    try:
        target_day = datetime.datetime.now() - datetime.timedelta(days=4)
        end_date = target_day.strftime("%Y%m%d")
        start_date = (
            target_day - datetime.timedelta(days=2)
        ).strftime("%Y%m%d")

        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M,PRECTOTCORR&community=AG"
            f"&longitude={longitude}&latitude={latitude}"
            f"&start={start_date}&end={end_date}&format=JSON"
        )

        response = requests.get(url, timeout=12)
        response.raise_for_status()
        data = response.json()

        temps = data["properties"]["parameter"]["T2M"]
        rain = data["properties"]["parameter"]["PRECTOTCORR"]
        latest_date = list(temps.keys())[-1]

        temperature = temps[latest_date]
        rainfall = rain[latest_date]

        if temperature == -999 or rainfall == -999:
            raise ValueError("NASA data is not available yet.")

        return temperature, rainfall, "NASA POWER satellite data"

    except Exception:
        fallback = {
            "Mzuzu": (23.5, 1.2),
            "Mzimba": (25.0, 0.5),
            "Karonga": (29.0, 4.5),
            "Rumphi": (22.8, 1.0),
            "Nkhata Bay": (26.5, 5.2),
        }

        temperature, rainfall = fallback.get(district, (24.0, 1.5))
        return temperature, rainfall, "Estimated fallback values"


# =========================================================
# 4. SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "latest_diagnosis" not in st.session_state:
    st.session_state.latest_diagnosis = None


def show_brand():
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🌽</div>
            <div>
                <div class="brand-title">AgriWise</div>
                <div class="brand-subtitle">Smart insights. Healthier harvests.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. LOGIN AND SIGNUP PAGE
# =========================================================
if not st.session_state.logged_in:

    show_brand()

    left, right = st.columns(
        [1.2, 0.9],
        gap="large",
        vertical_alignment="center",
    )

    with left:
        st.markdown(
            '<div class="eyebrow">AI-powered agriculture for Malawi</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<h1 class="hero-title">Grow smarter.<br>Harvest better.</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <p class="hero-copy">
                Identify common maize leaf diseases, explore localized
                climate information and make more informed decisions
                throughout your farming journey.
            </p>
            """,
            unsafe_allow_html=True,
        )

        feature1, feature2 = st.columns(2)

        with feature1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">📷</div>
                    <div class="feature-title">AI crop diagnosis</div>
                    <div class="feature-copy">
                        Upload a maize leaf photo for an AI prediction.
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

        with feature2:
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
        with st.container(border=True):
            st.markdown(
                '<div class="panel-kicker">Welcome to AgriWise</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="panel-title">Your farm, better informed.</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="panel-copy">Sign in to continue, or create an account to get started.</div>',
                unsafe_allow_html=True,
            )

            login_tab, signup_tab = st.tabs(
                ["Sign in", "Create account"]
            )

            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("Email address")
                    password = st.text_input(
                        "Password",
                        type="password",
                    )

                    login_button = st.form_submit_button(
                        "Sign in to AgriWise",
                        use_container_width=True,
                    )

                if login_button:
                    if not email.strip() or not password:
                        st.warning("Please enter your email and password.")
                    else:
                        user = verify_farmer(email.strip(), password)

                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user_name = user
                            st.rerun()
                        else:
                            st.error("Incorrect email address or password.")

            with signup_tab:
                with st.form("signup_form"):
                    full_name = st.text_input("Full name")
                    new_email = st.text_input("Email address")
                    new_password = st.text_input(
                        "Create password",
                        type="password",
                    )

                    signup_button = st.form_submit_button(
                        "Create my account",
                        use_container_width=True,
                    )

                if signup_button:
                    if not full_name.strip() or not new_email.strip() or not new_password:
                        st.warning("Please complete all fields.")
                    else:
                        created = add_farmer(
                            full_name.strip(),
                            new_email.strip(),
                            new_password,
                        )

                        if created:
                            st.success(
                                "Account created successfully. Please sign in."
                            )
                        else:
                            st.error(
                                "This email may already be registered."
                            )

    st.markdown(
        """
        <p style="text-align:center;color:rgba(255,255,255,.75);
        font-size:.8rem;margin-top:2rem;">
            AgriWise · Supporting informed maize farming decisions
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. FARMER DASHBOARD
# =========================================================
else:

    with st.sidebar:
        show_brand()
        st.markdown("---")
        st.markdown(f"### Welcome, {st.session_state.user_name}")
        st.caption("Your farming workspace")
        st.markdown("---")

        st.markdown("**Platform overview**")
        st.write("🌽 Supported crop: Maize")
        st.write("🔬 Disease categories: 4")
        st.write("🌦️ Climate districts: 5")

        st.markdown("---")

        if st.button("↩ Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.latest_diagnosis = None
            st.rerun()

    show_brand()

    st.markdown(
        '<div class="eyebrow">Your farming dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 class="dashboard-heading">Welcome to your farm dashboard.</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="dashboard-copy">Choose a service to check crop health or explore climate information for your area.</p>',
        unsafe_allow_html=True,
    )

    diagnosis_tab, climate_tab = st.tabs(
        ["📷 Crop diagnosis", "🌦️ Climate advisory"]
    )

    # -----------------------------------------------------
    # CROP DIAGNOSIS TAB
    # -----------------------------------------------------
    with diagnosis_tab:
        st.markdown(
            '<div class="section-label">Maize health assistant</div>',
            unsafe_allow_html=True,
        )

        upload_col, tips_col = st.columns([1.4, 0.8], gap="large")

        with upload_col:
            with st.container(border=True):
                st.subheader("Upload a maize leaf image")
                st.write(
                    "Choose a clear, close-up photo of one maize leaf. "
                    "Good lighting helps the model analyse visible symptoms."
                )

                uploaded_file = st.file_uploader(
                    "Choose JPG or PNG image",
                    type=["jpg", "jpeg", "png"],
                    key="maize_upload",
                )

                if uploaded_file is not None:
                    try:
                        image = Image.open(uploaded_file).convert("RGB")
                        st.image(
                            image,
                            caption="Selected maize leaf image",
                            use_container_width=True,
                        )

                        if st.button(
                            "🔍 Analyse image",
                            type="primary",
                            use_container_width=True,
                        ):
                            if not basic_leaf_screening(image):
                                st.session_state.latest_diagnosis = None
                                st.error(
                                    "The image did not pass the basic plant-image "
                                    "screening. Please try a clearer maize leaf photo."
                                )
                            else:
                                with st.spinner("Analysing your image..."):
                                    model = load_model()
                                    resized = image.resize((128, 128))
                                    array = np.asarray(resized) / 255.0
                                    array = np.expand_dims(array, axis=0)
                                    prediction = model.predict(array, verbose=0)

                                index = int(np.argmax(prediction[0]))
                                result = CLASS_NAMES[index]
                                confidence = float(prediction[0][index]) * 100

                                st.session_state.latest_diagnosis = {
                                    "result": result,
                                    "confidence": confidence,
                                }

                    except Exception as error:
                        st.error(f"Image analysis failed: {error}")

                diagnosis = st.session_state.latest_diagnosis

                if diagnosis:
                    result = diagnosis["result"]
                    confidence = diagnosis["confidence"]

                    st.divider()
                    st.subheader("Diagnosis result")

                    if result == "maize_healthy":
                        st.success(f"✅ {DISPLAY_NAMES[result]}")
                    else:
                        st.warning(f"⚠️ Possible {DISPLAY_NAMES[result]}")

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )
                    st.caption(f"Model confidence: {confidence:.1f}%")

                    st.info(
                        "This is an AI screening result, not a confirmed diagnosis. "
                        "Consult an agricultural extension officer before applying treatment."
                    )

        with tips_col:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-icon">☀️</div>
                    <div class="feature-title">Use natural light</div>
                    <div class="feature-copy">
                        Make sure the leaf is clearly visible.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🍃</div>
                    <div class="feature-title">Focus on one leaf</div>
                    <div class="feature-copy">
                        Keep the maize leaf as the main subject.
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📸</div>
                    <div class="feature-title">Avoid blurry photos</div>
                    <div class="feature-copy">
                        Hold the camera steady and take a close-up image.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # CLIMATE ADVISORY TAB
    # -----------------------------------------------------
    with climate_tab:
        st.markdown(
            '<div class="section-label">District climate information</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="weather-banner">
                <h2>Localized climate advisory</h2>
                <p>
                    View recent temperature and rainfall information
                    for selected districts in Malawi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        district = st.selectbox(
            "Select district / area",
            ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"],
        )

        coordinates = {
            "Mzuzu": (-11.46, 34.02),
            "Mzimba": (-11.90, 33.60),
            "Karonga": (-9.93, 33.93),
            "Rumphi": (-11.02, 33.85),
            "Nkhata Bay": (-11.60, 34.30),
        }

        latitude, longitude = coordinates[district]

        with st.spinner(f"Retrieving information for {district}..."):
            temperature, rainfall, source = get_weather(
                latitude,
                longitude,
                district,
            )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
        metric2.metric("🌧️ Rainfall", f"{rainfall:.1f} mm")
        metric3.metric("📍 Selected area", district)

        st.caption(f"Data source: {source}.")

        if source == "Estimated fallback values":
            st.warning(
                "Satellite data could not be retrieved. These are fallback "
                "estimates, not live weather observations."
            )

        if district in ["Karonga", "Nkhata Bay"]:
            st.info(
                "Please monitor official weather updates and local flood advisories."
            )
        elif district in ["Mzimba", "Rumphi"]:
            st.info(
                "Monitor soil moisture and follow local agricultural extension advice."
            )
        else:
            st.success(
                "No district-specific alert is configured for this area in the prototype."
            )

        st.caption(
            "This prototype's climate information is for guidance only and "
            "does not replace official forecasts or disaster-management alerts."
        )

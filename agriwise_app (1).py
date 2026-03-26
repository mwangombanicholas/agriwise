
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('agriwise_model.h5')

class_names = ['maize_cercospora_leaf_spot', 'maize_common_rust', 'maize_healthy', 'maize_northern_leaf_blight']
display_names = {
    'maize_cercospora_leaf_spot': 'Maize Cercospora Leaf Spot',
    'maize_common_rust': 'Maize Common Rust',
    'maize_healthy': 'Healthy Maize',
    'maize_northern_leaf_blight': 'Maize Northern Leaf Blight'
}

treatment_info = {
    'healthy': {'chemical': 'No treatment needed', 'organic': 'Continue good practices', 'prevention': 'Regular monitoring'},
    'common_rust': {'chemical': 'Azoxystrobin fungicide (K8,000-12,000/ha)', 'organic': 'Neem oil spray', 'prevention': 'Plant resistant varieties'},
    'cercospora_leaf_spot': {'chemical': 'Mancozeb fungicide (K5,000-8,000/kg)', 'organic': 'Remove infected leaves', 'prevention': 'Crop rotation'},
    'northern_leaf_blight': {'chemical': 'Azoxystrobin + Propiconazole (K10,000-15,000/ha)', 'organic': 'Baking soda solution', 'prevention': 'Use resistant varieties'}
}

st.title("🌾 AgriWise")
st.markdown("### AI-Powered Crop Health Advisor for Malawian Farmers")
st.markdown("**Supported crop:** 🌽 Maize (Corn) | **Model Accuracy:** 96%")

with st.sidebar:
    st.header("🌱 About AgriWise")
    st.write("Built by **Mzuzu University Students** for smallholder farmers.")
    st.write("**Detects:** Healthy, Common Rust, Cercospora Leaf Spot, Northern Leaf Blight")
    st.caption("© 2026 AgriWise | MZUNI SAVE Project")

tab1, tab2 = st.tabs(["📸 Disease Detection", "⛅ Weather Advisory"])

with tab1:
    st.header("Upload a photo of your maize leaf")
    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=250)

        with st.spinner("Analyzing..."):
            model = load_model()
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            idx = np.argmax(predictions[0])
            confidence = predictions[0][idx] * 100
            result = class_names[idx]
            display = display_names[result]

        if 'healthy' in result:
            st.success(f"### ✅ Diagnosis: {display}")
        else:
            st.warning(f"### ⚠️ Diagnosis: {display}")
        st.info(f"**Confidence:** {confidence:.1f}%")

        key = result.replace('maize_', '')
        info = treatment_info.get(key, treatment_info['healthy'])

        st.subheader("💊 Treatment")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Chemical:** {info['chemical']}")
        with col2:
            st.write(f"**Organic:** {info['organic']}")
        st.write(f"**Prevention:** {info['prevention']}")

with tab2:
    st.header("Weather Advisory")
    district = st.selectbox("Select district", ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"])
    if district in ["Karonga", "Nkhata Bay"]:
        st.error("🚨 FLOOD ALERT: Heavy rains expected")
    elif district == "Mzimba":
        st.warning("🌵 DRY SPELL ALERT: Irrigate if possible")
    else:
        st.success("✅ No active alerts")
    st.info("📅 Maize planting: November-December")
    st.success("🌽 Harvest: April-June")

st.caption("AgriWise | Mzuzu University | 96% Accuracy")

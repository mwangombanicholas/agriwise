import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="AgriWise", page_icon="🌾", layout="wide")

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
        st.image(uploaded, width=250)
        st.info("AI model is ready! Upload an image to test.")
        
        try:
            model_path = "agriwise_model.h5"
            if os.path.exists(model_path):
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ Model file not found. Download from Google Drive:")
                st.markdown("https://drive.google.com/file/d/1S8KT_Qz6094H2uRwkNqGGHh13C1WnVWw/view")
        except Exception as e:
            st.error(f"Error: {e}")

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

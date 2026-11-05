                </div>
                <div class="feature-card">
                    <div class="feature-icon">🍃</div>
                    <div class="feature-title">One leaf at a time</div>
                    <div class="feature-copy">Keep the maize leaf as the main subject in the frame.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📸</div>
                    <div class="feature-title">Keep it in focus</div>
                    <div class="feature-copy">Avoid motion blur, distant shots and heavily obstructed leaves.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">ℹ️</div>
                    <div class="feature-title">Use results carefully</div>
                    <div class="feature-copy">AI predictions can be wrong. Seek local expert advice before treatment.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # CLIMATE ADVISORY
    # -----------------------------------------------------
    with climate_tab:
        st.markdown('<div class="section-label">District climate information</div>', unsafe_allow_html=True)

        if video_b64:
            st.markdown(
                f"""
                <div class="weather-banner">
                    <h2>🌦️ Localized climate advisory</h2>
                    <p>Explore recent temperature and rainfall information for selected districts in Malawi.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="weather-banner">
                    <h2>🌦️ Localized climate advisory</h2>
                    <p>Explore recent temperature and rainfall information for selected districts in Malawi.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        district_col, detail_col = st.columns([1, 2], gap="large")
        with district_col:
            with st.container(border=True):
                st.markdown("### Choose your area")
                district = st.selectbox(
                    "District / area",
                    ["Mzuzu", "Mzimba", "Karonga", "Rumphi", "Nkhata Bay"],
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

        with st.spinner(f"Retrieving climate information for {district}..."):
            temperature, rainfall, data_source = get_real_weather(
                latitude, longitude, district
            )

        with detail_col:
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
            metric2.metric("🌧️ Rainfall", f"{rainfall:.1f} mm")
            metric3.metric("📍 Selected area", district)

        st.caption(f"Data source: {data_source}. NASA POWER data may be delayed; fallback values are estimates.")

        if district in ["Karonga", "Nkhata Bay"]:
            st.warning(
                "Flood-risk notice: monitor official weather updates and local "
                "advisories, especially in flood-prone areas."
            )
        elif district in ["Mzimba", "Rumphi"]:
            st.info(
                "Dry-spell notice: monitor soil moisture and follow local "
                "agricultural extension advice."
            )
        else:
            st.success(
                "No district-specific alert is configured for this area in the current prototype."
            )

        st.caption(
            "Climate figures and prototype alerts are informational. They should "
            "not replace official forecasts or local disaster-management guidance."
        )

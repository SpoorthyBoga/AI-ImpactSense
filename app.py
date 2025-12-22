import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Earthquake Impact Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD RESOURCES (Cached)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        with open("processed_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("advanced_models.pkl", "rb") as f:
            models = pickle.load(f)
        return data["scaler"], data["encoders"], models["reg"], models["class"]
    except FileNotFoundError:
        st.error("❌ Critical files missing! Please run '1_preprocess.py' and '3_train_advanced.py' first.")
        return None, None, None, None

scaler, encoders, reg_model, class_model = load_resources()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_energy_context(energy):
    tnt_tons = energy / (4.184 * 10**9)
    hiroshima = energy / (6.3 * 10**13)
    return tnt_tons, hiroshima

def get_damage_report(intensity):
    if intensity < 1.5:
        return "Negligible", "Felt only by a few. No structural damage."
    elif intensity < 2.5:
        return "Light", "Cosmetic damage (plaster cracks). Hanging objects swing."
    elif intensity < 3.5:
        return "Moderate", "Chimneys may fall. Unreinforced masonry at risk."
    else:
        return "Severe", "Structural collapse likely. Bridges/Roads impassable."

def get_risk_color(risk):
    colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Severe': 'darkred'}
    return colors.get(risk, 'blue')

# ==========================================
# 4. INITIALIZE SESSION STATE
# ==========================================
# This ensures data persists even after the button click resets
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = {}

# ==========================================
# 5. SIDEBAR - USER INPUTS
# ==========================================
with st.sidebar:
    st.header("🌍 Input Parameters")
    
    region = st.selectbox("Region", ["New Delhi", "Mumbai", "Kolkata", "Chennai", "Other"])
    
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=28.61, format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=77.20, format="%.4f")
        
    mag = st.slider("Magnitude (Richter)", 0.0, 10.0, 6.0, 0.1)
    depth = st.slider("Depth (km)", 0.0, 700.0, 10.0, 1.0)
    
    date_val = st.date_input("Date", datetime.now())
    time_val = st.time_input("Time", datetime.now())
    
    # Button triggers calculation
    if st.button("Predict Impact", type="primary", width="stretch"):
        if scaler:
            # --- Feature Engineering ---
            energy = 10 ** (1.5 * mag + 4.8)
            is_shallow = 1 if depth < 70 else 0
            is_night = 1 if (time_val.hour < 6 or time_val.hour > 20) else 0
            
            month = date_val.month
            season = "Winter" if month in [12,1,2] else "Summer" if month in [4,5,6] else "Monsoon"

            # Encoding
            try: region_code = encoders['region'].transform([region])[0]
            except: region_code = 0
            try: season_code = encoders['season'].transform([season])[0]
            except: season_code = 0

            # Prepare DataFrame
            input_data = pd.DataFrame([{
                "Latitude": lat, "Longitude": lon, "Depth": depth, "Magnitude": mag,
                "Energy": energy, "Is_Shallow": is_shallow, "Is_Night": is_night,
                "Region_Code": region_code, "Season_Code": season_code
            }])

            # Scaling
            scale_cols = ["Latitude", "Longitude", "Depth", "Magnitude", "Energy"]
            input_data[scale_cols] = scaler.transform(input_data[scale_cols])

            # Prediction
            features = [
                "Latitude", "Longitude", "Depth", "Magnitude",
                "Energy", "Is_Shallow", "Is_Night", 
                "Region_Code", "Season_Code"
            ]
            
            pred_intensity = reg_model.predict(input_data[features])[0]
            pred_risk_idx = class_model.predict(input_data[features])[0]
            pred_risk = encoders['risk'].inverse_transform([pred_risk_idx])[0]

            # Save to Session State
            st.session_state['results'] = {
                'intensity': pred_intensity,
                'risk': pred_risk,
                'energy': energy,
                'lat': lat,
                'lon': lon,
                'mag': mag
            }
            st.session_state['prediction_made'] = True

# ==========================================
# 6. MAIN DISPLAY LOGIC
# ==========================================
if st.session_state['prediction_made']:
    # Retrieve results from state
    res = st.session_state['results']
    pred_intensity = res['intensity']
    pred_risk = res['risk']
    energy = res['energy']
    lat = res['lat']
    lon = res['lon']
    mag = res['mag']

    # --- HEADER METRICS ---
    st.title("Prediction Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Impact Intensity Index", f"{pred_intensity:.4f}")
    m2.metric("Risk Classification", pred_risk)
    m3.metric("Energy Release (Joules)", f"{energy:.2e}")

    st.divider()

    # --- DETAILED CONTEXT ---
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Safety & Damage Report")
        
        # Alert Box
        risk_color = get_risk_color(pred_risk)
        if pred_risk == "Low":
            st.success(f"**Risk Level: {pred_risk}** - Minimal threat detected.")
        elif pred_risk == "Moderate":
            st.warning(f"**Risk Level: {pred_risk}** - Caution advised.")
        else:
            st.error(f"**Risk Level: {pred_risk}** - IMMEDIATE ACTION REQUIRED.")

        # Estimates
        dmg_title, dmg_desc = get_damage_report(pred_intensity)
        st.markdown(f"""
        **Infrastructure Impact:** {dmg_title}
        * {dmg_desc}
        """)

        # Energy
        tnt, hiro = get_energy_context(energy)
        st.markdown("**Energy Equivalence:**")
        st.caption(f"💣 Equivalent to **{tnt:,.0f} tons** of TNT")
        if hiro > 0.01:
            st.caption(f"☢️ Approx **{hiro:.4f}x** Hiroshima Atomic Bombs")

    with c2:
        st.subheader("🗺️ Impact Zone")
        
        # Folium Map
        m = folium.Map(location=[lat, lon], zoom_start=6, tiles="OpenStreetMap")
        
        folium.Circle(
            location=[lat, lon],
            radius=(mag**2)*1500,
            color=risk_color, fill=True, fill_opacity=0.3
        ).add_to(m)
        
        folium.Marker(
            [lat, lon],
            tooltip=f"Mag: {mag}",
            icon=folium.Icon(color=risk_color, icon="info-sign")
        ).add_to(m)
        
        st_folium(m, width=700, height=350)

    # --- PROTOCOLS ---
    with st.expander("🛡️ View Emergency Safety Protocols", expanded=True):
        if pred_risk == "Low":
            st.write("✅ **Monitor:** Keep updated via local radio/TV.")
            st.write("✅ **Report:** Report any minor cracks to authorities.")
        elif pred_risk == "Moderate":
            st.write("⚠️ **Inspect:** Check gas, water, and electric lines.")
            st.write("⚠️ **Prepare:** Secure heavy furniture; prepare for aftershocks.")
        elif pred_risk == "High":
            st.write("🚨 **EVACUATE:** Leave compromised buildings immediately.")
            st.write("🚨 **Avoid:** Stay away from glass, power lines, and trees.")
        else:
            st.write("🆘 **SHELTER:** If unable to evacuate, Drop, Cover, and Hold On.")
            st.write("🆘 **RESCUE:** Do not enter damaged zones. Wait for professional rescue.")

else:
    # Default State
    st.info("👈 Adjust parameters in the sidebar and click **Predict Impact** to start.")
    st.subheader("Training Data Insights")
    # Using 'width' instead of 'use_container_width' to prevent warnings
    st.image("visuals_preprocessing/3_geo_distribution.png", caption="Historical Earthquake Data", width="stretch")

import pickle
import pandas as pd
import numpy as np
import folium
import webbrowser
import os
import warnings

# Suppress sklearn warnings for clean output
warnings.filterwarnings('ignore')

# Files configuration
MODEL_FILE = "advanced_models.pkl"
DATA_FILE = "processed_data.pkl"
MAP_FILE = "impact_map.html"

class EarthquakeSystem:
    def __init__(self):
        print("\n" + "="*60)
        print("⚡  EARTHQUAKE IMPACT PREDICTION SYSTEM  ⚡")
        print("="*60)
        self.load_resources()

    def load_resources(self):
        print("📂 Loading Models & Encoders...")
        try:
            with open(DATA_FILE, "rb") as f:
                data = pickle.load(f)
                self.scaler = data["scaler"]
                self.encoders = data["encoders"]
            
            with open(MODEL_FILE, "rb") as f:
                models = pickle.load(f)
                self.reg_model = models["reg"]
                self.class_model = models["class"]
            print("✅ System Ready!\n")
                
        except FileNotFoundError:
            print(f"❌ Error: Critical files ('{DATA_FILE}' or '{MODEL_FILE}') missing.")
            print("   Please run '1_preprocess.py' and '3_train_advanced.py' first.")
            exit()

    def interpret_energy(self, energy_joules):
        """Converts abstract energy to relatable comparisons."""
        tnt_tons = energy_joules / (4.184 * 10**9)
        hiroshima = energy_joules / (6.3 * 10**13)
        
        print("\n💥 ENERGY RELEASE CONTEXT:")
        print(f"   • {energy_joules:.2e} Joules")
        print(f"   • Equivalent to {tnt_tons:,.0f} tons of TNT")
        if hiroshima > 0.01:
            print(f"   • ~{hiroshima:.4f}x the energy of the Hiroshima Atomic Bomb")

    def estimate_damage(self, intensity):
        """Heuristic damage estimation based on predicted intensity."""
        print("\n🏚️  ESTIMATED INFRASTRUCTURE IMPACT:")
        if intensity < 1.5:
            print("   • Buildings: Negligible damage.")
            print("   • Sensation: Felt only by a few people at rest.")
        elif intensity < 2.5:
            print("   • Buildings: Light cosmetic damage (plaster cracks).")
            print("   • Objects: Hanging objects may swing; windows may rattle.")
        elif intensity < 3.5:
            print("   • Buildings: Moderate damage. Chimneys may fall.")
            print("   • Risk: Unreinforced masonry at risk of partial collapse.")
        else:
            print("   • Buildings: SEVERE structural damage likely.")
            print("   • Infrastructure: Bridges/Roads may be impassable.")
            print("   • Secondary: High risk of landslides or liquefaction.")

    def get_protocols(self, risk):
        protocols = {
            "Low": ["✅ Minimal Danger", "• Monitor local news", "• Report minor damage"],
            "Moderate": ["⚠️ Caution", "• Inspect gas/water lines", "• Prepare for aftershocks"],
            "High": ["🚨 DANGER", "• Evacuate damaged buildings", "• Move to open ground away from glass"],
            "Severe": ["🆘 CRITICAL EMERGENCY", "• Deploy Search & Rescue", "• Shut down power/gas grids"]
        }
        return protocols.get(risk, ["Unknown Risk Level"])

    def generate_map(self, input_data, risk):
        lat, long = input_data['Latitude'], input_data['Longitude']
        mag = input_data['Magnitude']
        
        colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Severe': 'darkred'}
        color = colors.get(risk, 'blue')
        
        # Create base map
        m = folium.Map(location=[lat, long], zoom_start=6, tiles="OpenStreetMap")
        
        # Rich Popup Content
        info_html = f"""
        <div style="font-family: sans-serif; width: 200px;">
            <h4 style="margin-bottom:5px; border-bottom:1px solid #ccc;">Earthquake Alert</h4>
            <b>Region:</b> {input_data['Region']}<br>
            <b>Magnitude:</b> {mag}<br>
            <b>Depth:</b> {input_data['Depth']} km<br>
            <b>Time:</b> {input_data['Origin Time']}<br>
            <br>
            <b>Risk Level:</b> <span style="color:{color}; font-size:14px; font-weight:bold;">{risk}</span>
        </div>
        """
        
        # Add visual radius (Heuristic: Magnitude^2 * 1.5 km)
        folium.Circle(
            location=[lat, long], radius=(mag**2)*1500,
            color=color, fill=True, fill_opacity=0.3
        ).add_to(m)
        
        # Add Marker
        folium.Marker(
            [lat, long],
            popup=folium.Popup(info_html, max_width=250),
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
        
        m.save(MAP_FILE)
        print(f"\n🗺️  Interactive Map Generated: '{MAP_FILE}'")
        # webbrowser.open(MAP_FILE) # Uncomment to auto-open in browser

    def predict(self, input_data):
        print("-" * 60)
        print(f"🔎 ANALYZING NEW EVENT: {input_data['Region']}")
        print("-" * 60)

        # 1. Feature Engineering (On-the-fly)
        df = pd.DataFrame([input_data])
        energy_val = 10 ** (1.5 * df["Magnitude"] + 4.8)
        df["Energy"] = energy_val
        df["Is_Shallow"] = (df["Depth"] < 70).astype(int)
        
        dt = pd.to_datetime(df['Origin Time'])
        month = dt.dt.month.iloc[0]
        hour = dt.dt.hour.iloc[0]
        df["Is_Night"] = 1 if (hour < 6 or hour > 20) else 0
        
        season = "Winter" if month in [12,1,2] else "Summer" if month in [4,5,6] else "Monsoon"

        # 2. Encoding (Handle unseen labels gracefully)
        try: df['Region_Code'] = self.encoders['region'].transform([df['Region']])[0]
        except: df['Region_Code'] = 0 # Default to 0 if unknown region
            
        try: df['Season_Code'] = self.encoders['season'].transform([season])[0]
        except: df['Season_Code'] = 0

        # 3. Scaling
        scale_cols = ["Latitude", "Longitude", "Depth", "Magnitude", "Energy"]
        df[scale_cols] = self.scaler.transform(df[scale_cols])

        # 4. Feature Selection (Ensure correct order)
        features = [
            "Latitude", "Longitude", "Depth", "Magnitude",
            "Energy", "Is_Shallow", "Is_Night", 
            "Region_Code", "Season_Code"
        ]
        
        # 5. Prediction
        intensity_score = self.reg_model.predict(df[features])[0]
        risk_idx = self.class_model.predict(df[features])[0]
        risk_label = self.encoders['risk'].inverse_transform([risk_idx])[0]

        # 6. Display Results
        print(f"📊 PREDICTED INTENSITY INDEX: {intensity_score:.4f}")
        print(f"⚠️  PREDICTED RISK LEVEL:    {risk_label}")
        
        self.interpret_energy(energy_val.iloc[0])
        self.estimate_damage(intensity_score)
        
        print(f"\n📋 SAFETY PROTOCOLS ({risk_label}):")
        for p in self.get_protocols(risk_label):
            print(f"   {p}")

        self.generate_map(input_data, risk_label)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    system = EarthquakeSystem()
    
    # --- Test Case 1: High Impact ---
    high_impact_quake = {
        'Latitude': 28.61, 'Longitude': 77.20, 'Depth': 10.0, 
        'Magnitude': 7.5, 'Origin Time': '2023-12-25 04:30:00', 
        'Region': 'New Delhi'
    }
    system.predict(high_impact_quake)

    # --- Test Case 2: Low Impact ---
    low_impact_quake = {
        'Latitude': 19.07, 'Longitude': 72.87, 'Depth': 50.0, 
        'Magnitude': 3.2, 'Origin Time': '2024-01-10 14:00:00', 
        'Region': 'Mumbai'
    }
    system.predict(low_impact_quake)

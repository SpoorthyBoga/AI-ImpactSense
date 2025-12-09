import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATA_FILE = "India_Earthquake_Cleaned.xlsx"
OUTPUT_FILE = "processed_data.pkl"
IMG_DIR = "visuals_preprocessing"
os.makedirs(IMG_DIR, exist_ok=True)

def load_and_engineer_features():
    print(f"📦 Loading {DATA_FILE}...")
    try:
        df = pd.read_excel(DATA_FILE)
    except FileNotFoundError:
        print("❌ File not found. Please add the Excel file to this folder.")
        return None

    # --- 1. Cleaning ---
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    df = df.drop_duplicates().reset_index(drop=True)

    # --- 2. Feature Engineering ---
    print("⚙️  Engineering Features...")
    
    # Time Features
    df["Origin Time"] = pd.to_datetime(df["Origin Time"], errors="coerce")
    df["Year"] = df["Origin Time"].dt.year  # Added for visualization
    df["Month"] = df["Origin Time"].dt.month
    df["Hour"] = df["Origin Time"].dt.hour
    
    # Physics Features
    # Energy (Joules) = 10^(1.5*Mag + 4.8)
    df["Energy"] = 10 ** (1.5 * df["Magnitude"] + 4.8)
    df["Is_Shallow"] = (df["Depth"] < 70).astype(int)
    # Intensity Proxy
    df["Intensity_Index"] = df["Magnitude"] / np.log1p(df["Depth"] + 1)
    df["Is_Night"] = df["Hour"].apply(lambda x: 1 if (x < 6 or x > 20) else 0)
    
    # Categorical Features
    df["Region"] = df["Location"].apply(lambda x: x.split(",")[-1].strip() if pd.notna(x) else "Unknown")
    df["Season"] = df["Month"].apply(lambda m: "Winter" if m in [12,1,2] else "Summer" if m in [4,5,6] else "Monsoon")

    # --- 3. Encoding & Scaling ---
    encoders = {}
    encoders['region'] = LabelEncoder()
    df["Region_Code"] = encoders['region'].fit_transform(df["Region"])
    
    encoders['season'] = LabelEncoder()
    df["Season_Code"] = encoders['season'].fit_transform(df["Season"])

    # Define Targets
    y_reg = df["Intensity_Index"]
    
    # Risk Bins (Low, Moderate, High, Severe)
    bins = [-np.inf, 1.5, 2.5, 3.5, np.inf]
    labels = ['Low', 'Moderate', 'High', 'Severe']
    y_class_str = pd.cut(y_reg, bins=bins, labels=labels)
    
    encoders['risk'] = LabelEncoder()
    y_class = encoders['risk'].fit_transform(y_class_str)

    # Scaling
    scaler = StandardScaler()
    scale_cols = ["Latitude", "Longitude", "Depth", "Magnitude", "Energy"]
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Final Feature Set
    feature_cols = [
        "Latitude", "Longitude", "Depth", "Magnitude",
        "Energy", "Is_Shallow", "Is_Night", 
        "Region_Code", "Season_Code"
    ]
    X = df_scaled[feature_cols]

    # --- 4. Enhanced Visualizations ---
    print("🎨 Generating Enhanced Plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[feature_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.savefig(f"{IMG_DIR}/1_correlation_matrix.png")
    plt.close()

    # 2. Magnitude vs Depth (Scatter)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Magnitude', y='Depth', hue='Is_Shallow', palette='viridis', alpha=0.7)
    plt.title("Relationship: Magnitude vs Depth")
    plt.savefig(f"{IMG_DIR}/2_mag_vs_depth.png")
    plt.close()

    # 3. Geographic Distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Region', legend=False, palette='deep', s=30)
    plt.title("Geographic Distribution of Earthquakes")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"{IMG_DIR}/3_geo_distribution.png")
    plt.close()

    # 4. Target Distribution (Intensity Index) [NEW]
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Intensity_Index'], kde=True, color='teal', bins=30)
    plt.title("Distribution of Calculated Intensity Index")
    plt.xlabel("Intensity Index")
    plt.savefig(f"{IMG_DIR}/4_target_distribution.png")
    plt.close()

    # 5. Earthquake Counts by Year (Temporal Analysis) [NEW]
    plt.figure(figsize=(10, 5))
    if 'Year' in df.columns:
        sns.countplot(data=df, x='Year', palette='magma')
        plt.title("Earthquake Frequency by Year")
        plt.xticks(rotation=45)
        plt.savefig(f"{IMG_DIR}/5_temporal_trends.png")
    plt.close()

    # 6. Magnitude by Season (Boxplot) [NEW]
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Season', y='Magnitude', palette='Set2')
    plt.title("Magnitude Distribution by Season")
    plt.savefig(f"{IMG_DIR}/6_mag_by_season.png")
    plt.close()

    # 7. Risk Class Distribution [NEW]
    plt.figure(figsize=(8, 5))
    # Create a temporary series for plotting
    risk_counts = y_class_str.value_counts().sort_index()
    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette='Reds')
    plt.title("Distribution of Risk Classes")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.savefig(f"{IMG_DIR}/7_risk_distribution.png")
    plt.close()

    # --- 5. Save Data ---
    data_pack = {
        "X": X, "y_reg": y_reg, "y_class": y_class,
        "encoders": encoders, "scaler": scaler,
        "raw_df": df
    }
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data_pack, f)
    
    print(f"✅ Preprocessing Complete! Data saved to '{OUTPUT_FILE}'")
    print(f"   7 Enhanced Visuals saved in folder '{IMG_DIR}/'")

if __name__ == "__main__":
    load_and_engineer_features()
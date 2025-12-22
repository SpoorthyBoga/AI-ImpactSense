import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

INPUT_FILE = "processed_data.pkl"
OUTPUT_MODEL = "advanced_models.pkl"
IMG_DIR = "visuals_advanced"
os.makedirs(IMG_DIR, exist_ok=True)

def train_advanced():
    print("Training Advanced Ensemble Models...")
    
    # Load Data
    try:
        with open(INPUT_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError: return

    X, y_reg, y_class = data["X"], data["y_reg"], data["y_class"]
    le_risk = data["encoders"]["risk"]

    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    # --- Models Dictionary ---
    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoost": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42)
    }
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    # --- Training & Evaluation ---
    reg_scores = {}
    class_scores = {}
    class_preds = {}
    reg_preds = {}

    print(f"\n{'Model':<20} | {'R2 Score':<10} | {'Accuracy':<10}")
    print("-" * 45)

    for name in regressors.keys():
        # Regression
        regressors[name].fit(X_train, y_reg_train)
        y_reg_pred = regressors[name].predict(X_test)
        r2 = r2_score(y_reg_test, y_reg_pred)
        reg_scores[name] = r2
        reg_preds[name] = y_reg_pred
        
        # Classification
        classifiers[name].fit(X_train, y_class_train)
        y_class_pred = classifiers[name].predict(X_test)
        acc = accuracy_score(y_class_test, y_class_pred)
        class_scores[name] = acc
        class_preds[name] = y_class_pred
        
        print(f"{name:<20} | {r2:.4f}     | {acc:.4f}")

    # ==========================================
    # 3. ENHANCED VISUALIZATIONS
    # ==========================================
    print("\nGenerating Enhanced Plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Model Performance Bar Chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(reg_scores.keys(), reg_scores.values(), color=['teal', 'orange', 'purple'])
    axes[0].set_title("Regression Performance (R2)")
    axes[0].set_ylim(0, 1.1)
    
    axes[1].bar(class_scores.keys(), class_scores.values(), color=['teal', 'orange', 'purple'])
    axes[1].set_title("Classification Accuracy")
    axes[1].set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/1_model_comparison.png")
    plt.close()

    # --- Plot 2: Confusion Matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, pred) in enumerate(class_preds.items()):
        cm = confusion_matrix(y_class_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[i],
                    xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
        axes[i].set_title(f"{name} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/2_confusion_matrices.png")
    plt.close()

    # --- Plot 3: Feature Importance (Random Forest) [NEW] ---
    plt.figure(figsize=(10, 6))
    rf_model = classifiers["RandomForest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.barh(range(X.shape[1]), importances[indices], align="center", color="darkcyan")
    plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.savefig(f"{IMG_DIR}/3_feature_importance.png")
    plt.close()

    # --- Plot 4: Regression Residuals [NEW] ---
    # Residuals = Actual - Predicted. Closer to 0 line is better.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, pred) in enumerate(reg_preds.items()):
        residuals = y_reg_test - pred
        sns.scatterplot(x=y_reg_test, y=residuals, ax=axes[i], alpha=0.6)
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].set_title(f"{name} Residuals")
        axes[i].set_xlabel("Actual Intensity")
        axes[i].set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/4_regression_residuals.png")
    plt.close()

    # --- Plot 5: Predicted Class Distribution [NEW] ---
    plt.figure(figsize=(10, 6))
    pred_df = pd.DataFrame(class_preds)
    # Melt dataframe for seaborn boxplot
    pred_melt = pred_df.melt(var_name='Model', value_name='Predicted Class')
    sns.countplot(data=pred_melt, x='Predicted Class', hue='Model', palette='Set2')
    plt.title("Distribution of Predicted Risk Classes by Model")
    plt.xticks(ticks=range(len(le_risk.classes_)), labels=le_risk.classes_)
    plt.savefig(f"{IMG_DIR}/5_prediction_distribution.png")
    plt.close()

    # --- Save Best Models ---
    final_models = {
        "reg": regressors["RandomForest"],
        "class": classifiers["RandomForest"]
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(final_models, f)
        
    print(f"Advanced Models saved to '{OUTPUT_MODEL}'")
    print(f"   5 Enhanced Visuals saved in '{IMG_DIR}/'")

if __name__ == "__main__":
    train_advanced()
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report

INPUT_FILE = "processed_data.pkl"
OUTPUT_MODEL = "basic_models.pkl"
IMG_DIR = "visuals_basic"
os.makedirs(IMG_DIR, exist_ok=True)

def train_basic():
    print("🚀 Training Basic & SVM Models...")
    
    # Load Data
    try:
        with open(INPUT_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Data file missing. Run 1_preprocess.py first.")
        return

    X = data["X"]
    y_reg = data["y_reg"]
    y_class = data["y_class"]
    le_risk = data["encoders"]["risk"]

    # Train/Test Split
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    # ==========================================
    # 1. REGRESSION (Intensity Prediction)
    # ==========================================
    print("\n--- Regression Training ---")
    
    # A. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)
    y_reg_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_reg_test, y_reg_pred_lr)
    print(f"   📈 Linear Regression R2: {r2_lr:.4f}")

    # B. Support Vector Regressor (SVR)
    print("   ⏳ Training SVR (this might take a moment)...")
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_reg_train)
    y_reg_pred_svr = svr.predict(X_test)
    r2_svr = r2_score(y_reg_test, y_reg_pred_svr)
    print(f"   📈 SVM Regressor R2:     {r2_svr:.4f}")

    # ==========================================
    # 2. CLASSIFICATION (Risk Level)
    # ==========================================
    print("\n--- Classification Training ---")

    # A. Logistic Regression
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train, y_class_train)
    y_class_pred_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_class_test, y_class_pred_log)
    print(f"   📊 Logistic Regression Accuracy: {acc_log:.4f}")

    # B. Support Vector Classifier (SVC)
    print("   ⏳ Training SVC...")
    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_class_train)
    y_class_pred_svc = svc.predict(X_test)
    acc_svc = accuracy_score(y_class_test, y_class_pred_svc)
    print(f"   📊 SVM Classifier Accuracy:      {acc_svc:.4f}")

    # ==========================================
    # 3. VISUALIZATIONS
    # ==========================================
    print("\n🎨 Generating Visual Comparisons...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot A: Regression Comparison (Bar Chart) ---
    plt.figure(figsize=(8, 5))
    models = ['Linear Regression', 'SVR']
    scores = [r2_lr, r2_svr]
    bars = plt.bar(models, scores, color=['#4c72b0', '#55a868'])
    plt.title('Regression Model Comparison (R2 Score)')
    plt.ylabel('R2 Score')
    plt.ylim(0, 1.1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    plt.savefig(f"{IMG_DIR}/regression_comparison_bar.png")
    plt.close()

    # --- Plot B: Actual vs Predicted (Side-by-Side) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear Regression Plot
    axes[0].scatter(y_reg_test, y_reg_pred_lr, alpha=0.5, color='blue', edgecolors='k')
    axes[0].plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    axes[0].set_title(f"Linear Regression (R2={r2_lr:.2f})")
    axes[0].set_xlabel("Actual Intensity")
    axes[0].set_ylabel("Predicted")

    # SVR Plot
    axes[1].scatter(y_reg_test, y_reg_pred_svr, alpha=0.5, color='green', edgecolors='k')
    axes[1].plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    axes[1].set_title(f"SVM Regressor (R2={r2_svr:.2f})")
    axes[1].set_xlabel("Actual Intensity")
    axes[1].set_ylabel("Predicted")

    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/regression_actual_vs_pred.png")
    plt.close()

    # --- Plot C: Classification Comparison (Bar Chart) ---
    plt.figure(figsize=(8, 5))
    models = ['Logistic Regression', 'SVC']
    scores = [acc_log, acc_svc]
    bars = plt.bar(models, scores, color=['#8172b3', '#c44e52'])
    plt.title('Classification Model Comparison (Accuracy)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    plt.savefig(f"{IMG_DIR}/classification_comparison_bar.png")
    plt.close()

    # --- Plot D: Confusion Matrices (Side-by-Side) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Logistic Matrix
    cm_log = confusion_matrix(y_class_test, y_class_pred_log)
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
    axes[0].set_title(f"Logistic Regression (Acc={acc_log:.2f})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # SVC Matrix
    cm_svc = confusion_matrix(y_class_test, y_class_pred_svc)
    sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
    axes[1].set_title(f"SVM Classifier (Acc={acc_svc:.2f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/classification_confusion_matrices.png")
    plt.close()

    # --- Save Models ---
    models = {
        "lr": lr, 
        "svr": svr, 
        "log_reg": log_reg,
        "svc": svc
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(models, f)
    
    print(f"✅ All Basic & SVM models saved to '{OUTPUT_MODEL}'")
    print(f"   Visuals saved in '{IMG_DIR}/'")

if __name__ == "__main__":
    train_basic()
"""
Model training script for Academic Burnout Detection.
Trains a Random Forest Classifier and saves model + scaler artifacts.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "student_data.csv")
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "burnout_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = [
    "login_frequency",
    "avg_study_duration",
    "assignment_delay_days",
    "forum_participation",
    "quiz_score_avg",
    "resource_access_count",
    "missed_deadlines_pct",
    "session_break_freq",
]
TARGET_COL = "burnout_risk"
LABEL_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def train():
    # ── 1. generate data if absent ─────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("[INFO] Generating synthetic dataset ...")
        sys.path.insert(0, os.path.join(ROOT, "data"))
        from synthetic_data import generate_dataset
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"   Class distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # ── 2. split ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── 3. scale ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 4. train ───────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # ── 5. evaluate ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test Accuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values())))

    cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")

    # ── 6. feature importances ─────────────────────────────────────────────────
    fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature Importances:")
    print(fi.to_string())

    # ── 7. save ────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[SAVED] Model  -> {MODEL_PATH}")
    print(f"[SAVED] Scaler -> {SCALER_PATH}")


if __name__ == "__main__":
    train()

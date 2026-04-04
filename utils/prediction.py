"""
Prediction utilities for Academic Burnout Detection.
"""

import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "burnout_model.pkl")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.pkl")

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

LABEL_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
COLOR_MAP = {
    "Low Risk":    "#22c55e",   # green
    "Medium Risk": "#f59e0b",   # amber
    "High Risk":   "#ef4444",   # red
}
ICON_MAP = {
    "Low Risk":    "✅",
    "Medium Risk": "⚠️",
    "High Risk":   "🚨",
}

RECOMMENDATIONS = {
    "Low Risk": [
        "Keep up the consistent login and study habits!",
        "Consider peer mentoring to support classmates.",
        "Explore advanced/extra-credit resources to stay engaged.",
        "Maintain healthy break frequency between sessions.",
    ],
    "Medium Risk": [
        "Schedule a check-in with your academic advisor soon.",
        "Use the Pomodoro technique to reduce assignment delays.",
        "Join or revisit study groups for peer accountability.",
        "Aim to reduce missed deadlines by 50% over the next 2 weeks.",
        "Explore campus wellness programs.",
    ],
    "High Risk": [
        "🚨 Immediate counselor intervention is recommended.",
        "Request assignment deadline extensions where possible.",
        "Engage with campus mental health resources this week.",
        "Reduce study session length — focus on quality over quantity.",
        "Reach out to a trusted faculty member or peer.",
        "Consider a temporary reduced course load.",
    ],
}


def load_model():
    """Load model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python model/train.py` first."
        )
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def predict_single(features: dict) -> dict:
    """
    Predict burnout risk for a single student.

    Parameters
    ----------
    features : dict  — keys matching FEATURE_COLS

    Returns
    -------
    dict with keys: label, confidence, probabilities, recommendations
    """
    clf, scaler = load_model()
    X = np.array([[features[c] for c in FEATURE_COLS]])
    X_s = scaler.transform(X)
    pred = int(clf.predict(X_s)[0])
    proba = clf.predict_proba(X_s)[0]
    label = LABEL_MAP[pred]
    return {
        "label":           label,
        "confidence":      float(proba[pred]),
        "probabilities":   {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
        "color":           COLOR_MAP[label],
        "icon":            ICON_MAP[label],
        "recommendations": RECOMMENDATIONS[label],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict burnout risk for a DataFrame of students.

    Parameters
    ----------
    df : DataFrame — must contain FEATURE_COLS columns (and optionally student_id)

    Returns
    -------
    DataFrame with added burnout_risk_label and confidence columns
    """
    clf, scaler = load_model()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded CSV: {missing}")
    X = df[FEATURE_COLS].values
    X_s = scaler.transform(X)
    preds = clf.predict(X_s).astype(int)
    probas = clf.predict_proba(X_s)
    df = df.copy()
    df["burnout_risk_label"] = [LABEL_MAP[p] for p in preds]
    df["confidence"] = [round(probas[i][p] * 100, 1) for i, p in enumerate(preds)]
    return df


def get_feature_importances() -> pd.Series:
    clf, _ = load_model()
    return pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

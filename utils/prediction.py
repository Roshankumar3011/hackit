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


def get_model_metrics() -> dict:
    """
    Evaluate the saved model on the held-out test split.

    Replays the exact same train/test split used during training
    (test_size=0.20, random_state=42, stratify=y) so no retraining is needed.

    Returns
    -------
    dict with keys:
        accuracy   : float
        precision  : float  (macro)
        recall     : float  (macro)
        f1         : float  (macro)
        per_class  : dict   label -> {precision, recall, f1, support}
        conf_matrix: 2-D list (rows = true, cols = predicted)
        labels     : list of class label strings
    """
    import os, sys
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

    # ── generate data if absent ────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        sys.path.insert(0, os.path.join(ROOT_DIR, "data"))
        from synthetic_data import generate_dataset  # type: ignore
        os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].values
    y = df["burnout_risk"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf, scaler = load_model()
    X_test_s = scaler.transform(X_test)
    y_pred = clf.predict(X_test_s)

    labels_order = [0, 1, 2]
    label_names = [LABEL_MAP[k] for k in labels_order]

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec   = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1    = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm    = confusion_matrix(y_test, y_pred, labels=labels_order).tolist()

    report = classification_report(
        y_test, y_pred,
        labels=labels_order,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {name: report[name] for name in label_names}

    return {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "per_class":  per_class,
        "conf_matrix": cm,
        "labels":     label_names,
    }

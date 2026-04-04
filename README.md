# 🧠 BurnoutIQ — AI-Powered Academic Burnout Detection System

> A machine learning prototype that proactively identifies students at risk of mental exhaustion by analyzing behavioral and academic engagement metrics.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard (model trains automatically on first run)
streamlit run app.py
```

---

## 📁 Project Structure

```
hack/
├── app.py                        # Main Streamlit dashboard
├── requirements.txt
├── model/
│   ├── train.py                  # Model training script
│   ├── burnout_model.pkl         # Saved Random Forest (auto-generated)
│   └── scaler.pkl                # Feature scaler (auto-generated)
├── data/
│   ├── synthetic_data.py         # Synthetic dataset generator
│   └── student_data.csv          # Generated training data (auto-generated)
└── utils/
    ├── prediction.py             # Prediction helpers & recommendation engine
    └── visualizations.py         # Plotly chart functions
```

---

## 🧪 Features

| Feature | Description |
|---|---|
| **Live Prediction** | Real-time burnout risk from 8 behavioral sliders |
| **Batch Analysis** | Upload a CSV to analyze an entire student cohort |
| **Model Insights** | Feature importance chart and model architecture details |
| **Explainable AI** | Per-class probabilities and confidence scores |
| **Recommendations** | Actionable counselor guidance per risk level |
| **Privacy-First** | Anonymized student IDs, no PII stored |

---

## 📊 Input Metrics

| Metric | Range | Meaning |
|---|---|---|
| `login_frequency` | 0–14 | Logins per week |
| `avg_study_duration` | 0–8 hrs | Daily study time |
| `assignment_delay_days` | 0–10 days | Avg late submission days |
| `forum_participation` | 0–15 posts | Forum posts/week |
| `quiz_score_avg` | 0–100% | Average quiz score |
| `resource_access_count` | 0–25 | Resources opened/week |
| `missed_deadlines_pct` | 0–80% | % of deadlines missed |
| `session_break_freq` | 0–12 | Avg breaks/session |

---

## 🤖 Model

- **Algorithm**: `RandomForestClassifier` (scikit-learn)
- **Estimators**: 200 trees · Max Depth: 12
- **Classes**: Low Risk · Medium Risk · High Risk
- **Training Data**: 2,500 synthetic anonymized student records
- **Expected Accuracy**: ≥ 88% on held-out test set

---

## 🔒 Privacy Architecture

- Student records use anonymized IDs (`STU-XXXXX`)
- No name, email, or PII fields
- All processing is local — no external API calls
- Batch results downloadable as anonymized CSV

---

## 📋 Batch CSV Template

Download the template from the app, or follow this column structure:

```
student_id,login_frequency,avg_study_duration,assignment_delay_days,
forum_participation,quiz_score_avg,resource_access_count,
missed_deadlines_pct,session_break_freq
STU-00001,7,2.5,1.0,4,72.5,8,15.0,3.0
```

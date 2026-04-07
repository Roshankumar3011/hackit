"""
AI-Powered Academic Burnout Detection System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Streamlit Dashboard
"""

import os
import sys
import io
import pandas as pd
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.prediction import (
    predict_single,
    predict_batch,
    get_feature_importances,
    get_model_metrics,
    FEATURE_COLS,
    COLOR_MAP,
    ICON_MAP,
)
from utils.visualizations import (
    gauge_chart,
    probability_bar,
    feature_importance_chart,
    batch_risk_pie,
    batch_confidence_histogram,
    confusion_matrix_chart,
    per_class_metrics_chart,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BurnoutIQ — Academic Burnout Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── global ── */
  html, body, [data-testid="stAppViewContainer"] {
      background: #080f1a !important;
      color: #e2e8f0;
      font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
      background: #0d1827 !important;
      border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ── headings ── */
  h1, h2, h3 { font-family: 'Inter', sans-serif; }

  /* ── hero banner ── */
  .hero {
      background: linear-gradient(135deg, #0d1827 0%, #0f2d4a 50%, #0d1827 100%);
      border: 1px solid #1e3a5f;
      border-radius: 20px;
      padding: 2.5rem 2rem;
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
  }
  .hero::before {
      content: "";
      position: absolute;
      top: -60px; right: -60px;
      width: 220px; height: 220px;
      background: radial-gradient(circle, rgba(99,102,241,0.25), transparent 70%);
      border-radius: 50%;
  }
  .hero-title {
      font-size: 2.4rem;
      font-weight: 800;
      background: linear-gradient(90deg, #6366f1, #818cf8, #a5b4fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin: 0 0 0.4rem 0;
  }
  .hero-sub {
      font-size: 1rem;
      color: #94a3b8;
      margin: 0 0 1rem 0;
  }
  .badge {
      display: inline-block;
      background: rgba(99,102,241,0.15);
      border: 1px solid rgba(99,102,241,0.4);
      border-radius: 999px;
      padding: 4px 14px;
      font-size: 0.78rem;
      color: #a5b4fc;
      margin-right: 8px;
      margin-bottom: 4px;
  }

  /* ── metric cards ── */
  .metric-card {
      background: #0d1827;
      border: 1px solid #1e3a5f;
      border-radius: 14px;
      padding: 1.2rem 1.4rem;
      text-align: center;
      transition: border-color .25s;
  }
  .metric-card:hover { border-color: #6366f1; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #6366f1; }
  .metric-label { font-size: 0.82rem; color: #64748b; margin-top: 4px; }

  /* ── section headers ── */
  .section-header {
      font-size: 1.2rem;
      font-weight: 700;
      color: #e2e8f0;
      border-left: 4px solid #6366f1;
      padding-left: 0.8rem;
      margin: 1.8rem 0 1rem 0;
  }

  /* ── result alert ── */
  .result-card {
      border-radius: 16px;
      padding: 1.6rem 1.8rem;
      border: 1px solid;
      margin-top: 1rem;
      position: relative;
      overflow: hidden;
  }
  .result-title { font-size: 1.5rem; font-weight: 800; margin: 0 0 0.3rem 0; }
  .result-conf  { font-size: 0.9rem; color: #94a3b8; }

  /* ── recommendation item ── */
  .rec-item {
      background: #0d1827;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 0.7rem 1rem;
      margin-bottom: 0.5rem;
      font-size: 0.92rem;
      color: #cbd5e1;
  }

  /* ── input labels ── */
  label { color: #94a3b8 !important; font-size: 0.85rem !important; }

  /* ── tab style overrides ── */
  [data-testid="stTabs"] button {
      color: #64748b !important;
      font-weight: 500;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
      color: #6366f1 !important;
      border-bottom-color: #6366f1 !important;
  }

  /* ── scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1827; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

  /* ── divider ── */
  hr { border-color: #1e3a5f !important; }

  /* ── privacy notice ── */
  .privacy-box {
      background: rgba(99,102,241,0.06);
      border: 1px solid rgba(99,102,241,0.2);
      border-radius: 10px;
      padding: 0.8rem 1rem;
      font-size: 0.82rem;
      color: #94a3b8;
  }

  /* ── batch table ── */
  [data-testid="stDataFrame"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 BurnoutIQ")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        label="Go to",
        options=["🔍 Live Prediction", "📂 Batch Analysis", "📊 Model Insights", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div class='privacy-box'>
        🔒 <strong>Privacy First</strong><br>
        All predictions use anonymized student IDs.
        No personally identifiable information is stored or transmitted.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Model Info**
    - `RandomForestClassifier`
    - 200 estimators · max depth 12
    - 3 risk classes
    - Feature-scaled input
    """)
    st.markdown("---")
    st.caption("v1.0.0 · Built for Educators & Counselors")


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <p class="hero-title">🧠 BurnoutIQ</p>
    <p class="hero-sub">
        AI-Powered Academic Burnout Detection System —
        Proactively identify students at risk before it's too late.
    </p>
    <span class="badge">🔒 Privacy-First</span>
    <span class="badge">🤖 Random Forest AI</span>
    <span class="badge">⚡ Real-Time Predictions</span>
    <span class="badge">📊 Explainable AI</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: ensure model exists
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(ROOT, "model", "burnout_model.pkl")

@st.cache_resource(show_spinner="🔄 Training model (first-time setup) ...")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        from model.train import train
        train()
    return True

try:
    ensure_model()
except Exception as e:
    st.error(f"❌ Failed to load/train model: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "🔍 Live Prediction" in page:

    st.markdown("<div class='section-header'>📋 Student Engagement Metrics</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:-0.6rem;margin-bottom:1.2rem;'>"
        "Adjust the sliders below to reflect the student's recent activity. Predictions update instantly.</p>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        c1, c2 = st.columns(2)
        with c1:
            login_frequency = st.slider(
                "🔐 Login Frequency (per week)",
                min_value=0, max_value=14, value=7,
                help="Number of times the student logged into the LMS this week",
                key="login_freq",
            )
            assignment_delay = st.slider(
                "⏰ Assignment Delay (days avg)",
                min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                help="Average days late assignments are submitted",
                key="assign_delay",
            )
            quiz_score = st.slider(
                "📝 Quiz Score Average (%)",
                min_value=0.0, max_value=100.0, value=68.0, step=0.5,
                help="Average quiz/assessment score",
                key="quiz_score",
            )
            missed_deadlines = st.slider(
                "❌ Missed Deadlines (%)",
                min_value=0.0, max_value=80.0, value=20.0, step=0.5,
                help="Percentage of total deadlines missed",
                key="missed_dl",
            )

        with c2:
            study_duration = st.slider(
                "📚 Avg Study Duration (hrs/day)",
                min_value=0.0, max_value=8.0, value=2.5, step=0.1,
                help="Average daily study time in hours",
                key="study_dur",
            )
            forum_participation = st.slider(
                "💬 Forum Participation (posts/week)",
                min_value=0, max_value=15, value=3,
                help="Number of forum/discussion board posts per week",
                key="forum_part",
            )
            resource_access = st.slider(
                "📖 Resource Access Count (per week)",
                min_value=0, max_value=25, value=6,
                help="Number of learning resources opened this week",
                key="res_access",
            )
            break_freq = st.slider(
                "☕ Session Break Frequency",
                min_value=0.0, max_value=12.0, value=4.0, step=0.5,
                help="Average number of breaks taken during study sessions",
                key="break_freq",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "⚡ Analyze Burnout Risk",
            type="primary",
            use_container_width=True,
            key="predict_btn",
        )

    with col_r:
        features = {
            "login_frequency":       login_frequency,
            "avg_study_duration":    study_duration,
            "assignment_delay_days": assignment_delay,
            "forum_participation":   forum_participation,
            "quiz_score_avg":        quiz_score,
            "resource_access_count": resource_access,
            "missed_deadlines_pct":  missed_deadlines,
            "session_break_freq":    break_freq,
        }

        # Auto-predict on slider change
        result = predict_single(features)

        color  = result["color"]
        icon   = result["icon"]
        label  = result["label"]
        conf   = result["confidence"]
        recs   = result["recommendations"]
        probas = result["probabilities"]

        # ── result card ──
        st.markdown(f"""
        <div class="result-card" style="background:{'rgba(239,68,68,0.08)' if label=='High Risk' else 'rgba(245,158,11,0.08)' if label=='Medium Risk' else 'rgba(34,197,94,0.08)'}; border-color:{color}33;">
            <p style="font-size:2.8rem;margin:0;">{icon}</p>
            <p class="result-title" style="color:{color};">{label}</p>
            <p class="result-conf">Prediction confidence: <strong style="color:{color};">{conf*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # ── gauge ──
        st.plotly_chart(gauge_chart(conf, label, color), use_container_width=True, config={"displayModeBar": False})

        # ── probability breakdown ──
        st.markdown("<p style='color:#64748b;font-size:0.82rem;margin-bottom:4px;'>Risk class probabilities</p>", unsafe_allow_html=True)
        st.plotly_chart(probability_bar(probas), use_container_width=True, config={"displayModeBar": False})

    # ── recommendations ──
    st.markdown("<div class='section-header'>💡 Recommended Actions</div>", unsafe_allow_html=True)
    rec_cols = st.columns(min(len(recs), 3))
    for i, rec in enumerate(recs):
        with rec_cols[i % len(rec_cols)]:
            st.markdown(f"<div class='rec-item'>→ {rec}</div>", unsafe_allow_html=True)

    # ── quick stats ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📌 Current Input Summary</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("🔐 Logins/Week", login_frequency),
        ("📚 Study Hrs/Day", f"{study_duration:.1f}"),
        ("❌ Missed Deadlines", f"{missed_deadlines:.0f}%"),
        ("📝 Quiz Avg", f"{quiz_score:.1f}%"),
    ]
    for col, (lbl, val) in zip([m1, m2, m3, m4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "📂 Batch Analysis" in page:
    st.markdown("<div class='section-header'>📂 Batch Student Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:-0.5rem;margin-bottom:1rem;'>"
        "Upload a CSV of anonymized student records to receive risk classifications for the entire cohort.</p>",
        unsafe_allow_html=True,
    )

    # Template download
    template_data = {col: [0] for col in FEATURE_COLS}
    template_df = pd.DataFrame(template_data)
    template_df.insert(0, "student_id", ["STU-00001"])
    csv_bytes = template_df.to_csv(index=False).encode()

    dl_col, up_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="⬇️ Download CSV Template",
            data=csv_bytes,
            file_name="burnout_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with up_col:
        uploaded = st.file_uploader(
            "Upload student CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="batch_upload",
        )

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
            results_df = predict_batch(raw_df)

            st.success(f"✅ Analyzed **{len(results_df)}** student records")

            tab_table, tab_pie, tab_hist = st.tabs(["📋 Results Table", "🥧 Risk Distribution", "📊 Confidence Histogram"])

            with tab_table:
                risk_color_map = {
                    "Low Risk":    "background-color: rgba(34,197,94,0.12); color:#22c55e",
                    "Medium Risk": "background-color: rgba(245,158,11,0.12); color:#f59e0b",
                    "High Risk":   "background-color: rgba(239,68,68,0.12); color:#ef4444",
                }

                def color_row(row):
                    style = risk_color_map.get(row["burnout_risk_label"], "")
                    return [""] * (len(row) - 2) + [style, ""]

                styled = results_df.style.apply(color_row, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)

                export_csv = results_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=export_csv,
                    file_name="burnout_predictions.csv",
                    mime="text/csv",
                )

            with tab_pie:
                c1, c2, c3 = st.columns(3)
                risk_counts = results_df["burnout_risk_label"].value_counts()
                total = len(results_df)
                for col, (risk, color) in zip([c1, c2, c3], [
                    ("Low Risk", "#22c55e"), ("Medium Risk", "#f59e0b"), ("High Risk", "#ef4444")
                ]):
                    count = risk_counts.get(risk, 0)
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="border-color:{color}33;">
                            <div class="metric-value" style="color:{color};">{count}</div>
                            <div class="metric-label">{risk} ({count/total*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.plotly_chart(batch_risk_pie(results_df), use_container_width=True, config={"displayModeBar": False})

            with tab_hist:
                st.plotly_chart(batch_confidence_histogram(results_df), use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        # Show sample structure
        st.markdown("<div class='section-header'>📌 Expected CSV Columns</div>", unsafe_allow_html=True)
        col_info = {
            "Column": FEATURE_COLS,
            "Description": [
                "Logins per week (int)",
                "Avg daily study hours (float)",
                "Avg days late for assignments (float)",
                "Forum posts per week (int)",
                "Average quiz score 0-100 (float)",
                "Learning resources opened/week (int)",
                "% of deadlines missed (float)",
                "Avg breaks per study session (float)",
            ],
            "Example": ["7", "2.5", "1.0", "4", "72.5", "8", "15.0", "3.0"],
        }
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Insights" in page:
    st.markdown("<div class='section-header'>📊 Model Insights & Explainability</div>", unsafe_allow_html=True)

    try:
        importances = get_feature_importances()

        # Feature importance chart
        st.markdown(
            "<p style='color:#64748b;font-size:0.88rem;margin-top:-0.5rem;margin-bottom:1rem;'>"
            "Feature importance shows which behavioral metrics have the greatest influence on the model's predictions.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(feature_importance_chart(importances), use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-header'>🏋️ Model Architecture</div>", unsafe_allow_html=True)
            arch_data = {
                "Parameter":     ["Algorithm", "Estimators", "Max Depth", "Min Samples Split", "Min Samples Leaf", "Class Weight", "Training Set", "Features"],
                "Value":         ["Random Forest", "200 trees", "12", "5", "2", "Balanced", "80% of 2,500 records", "8 behavioral metrics"],
            }
            st.table(pd.DataFrame(arch_data))

        with c2:
            st.markdown("<div class='section-header'>🎯 Risk Level Definitions</div>", unsafe_allow_html=True)
            for risk, color, desc in [
                ("✅ Low Risk",    "#22c55e", "Student is engaged, meeting deadlines, and performing well. Continue monitoring; consider peer-mentoring roles."),
                ("⚠️ Medium Risk", "#f59e0b", "Showing early warning signs. Advisor check-in recommended within 1–2 weeks."),
                ("🚨 High Risk",   "#ef4444", "Significant disengagement detected. Immediate counselor intervention is strongly advised."),
            ]:
                st.markdown(f"""
                <div class="rec-item" style="border-color:{color}44; margin-bottom:0.8rem;">
                    <strong style="color:{color};">{risk}</strong><br>
                    <span style="color:#94a3b8;font-size:0.85rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='section-header'>📐 Feature Importance Table</div>", unsafe_allow_html=True)
        fi_df = importances.sort_values(ascending=False).reset_index()
        fi_df.columns = ["Feature", "Importance Score"]
        fi_df["Importance Score"] = fi_df["Importance Score"].round(4)
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df = fi_df[["Rank", "Feature", "Importance Score"]]
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Could not load model insights: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "📈 Model Performance" in page:
    st.markdown("<div class='section-header'>📈 Model Performance Metrics</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:-0.5rem;margin-bottom:1.2rem;'>"
        "Evaluated on the held-out 20% test split (same split used during training, random_state=42). "
        "No data snooping — the model was never trained on these samples.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("⚙️ Computing metrics from test set ..."):
        try:
            metrics = get_model_metrics()
        except Exception as e:
            st.error(f"❌ Could not compute metrics: {e}")
            st.stop()

    # ── top-level metric cards ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, (icon, label, val, color) in zip(
        [m1, m2, m3, m4],
        [
            ("🎯", "Accuracy",  metrics["accuracy"]  * 100, "#6366f1"),
            ("🔬", "Precision", metrics["precision"] * 100, "#22c55e"),
            ("📡", "Recall",    metrics["recall"]    * 100, "#f59e0b"),
            ("⚖️", "F1-Score",  metrics["f1"]        * 100, "#ec4899"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div class="metric-value" style="color:{color};">{val:.1f}%</div>
                <div class="metric-label">{label} (macro)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── two-column layout: confusion matrix + per-class chart ───────────────
    tab_cm, tab_pc, tab_tbl = st.tabs(
        ["🔲 Confusion Matrix", "📊 Per-Class Metrics", "📋 Detailed Report"]
    )

    with tab_cm:
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;'>"
            "Rows = True class, Columns = Predicted class. Cell colour intensity shows row-normalised rate."
            "</p>", unsafe_allow_html=True
        )
        st.plotly_chart(
            confusion_matrix_chart(metrics["conf_matrix"], metrics["labels"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab_pc:
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;'>"
            "Per-class Precision, Recall and F1-Score for each burnout risk category."
            "</p>", unsafe_allow_html=True
        )
        st.plotly_chart(
            per_class_metrics_chart(metrics["per_class"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab_tbl:
        st.markdown("<div class='section-header'>📋 Full Classification Report</div>", unsafe_allow_html=True)
        rows = []
        for lbl, vals in metrics["per_class"].items():
            rows.append({
                "Risk Class":  lbl,
                "Precision":   f"{vals['precision']*100:.1f}%",
                "Recall":      f"{vals['recall']*100:.1f}%",
                "F1-Score":    f"{vals['f1-score']*100:.1f}%",
                "Support":     int(vals['support']),
            })
        import pandas as _pd
        report_df = _pd.DataFrame(rows)

        def color_report(row):
            label = row["Risk Class"]
            colors = {"Low Risk": "rgba(34,197,94,0.10)", "Medium Risk": "rgba(245,158,11,0.10)", "High Risk": "rgba(239,68,68,0.10)"}
            bg = colors.get(label, "")
            return [f"background-color:{bg}" if bg else ""] * len(row)

        st.dataframe(
            report_df.style.apply(color_report, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        # Summary aggregate row
        agg1, agg2, agg3, agg4 = st.columns(4)
        for col, (lbl, val, color) in zip(
            [agg1, agg2, agg3, agg4],
            [
                ("Overall Accuracy",  f"{metrics['accuracy']*100:.2f}%",  "#6366f1"),
                ("Macro Precision",   f"{metrics['precision']*100:.2f}%", "#22c55e"),
                ("Macro Recall",      f"{metrics['recall']*100:.2f}%",    "#f59e0b"),
                ("Macro F1",          f"{metrics['f1']*100:.2f}%",        "#ec4899"),
            ],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color}44;">
                    <div class="metric-value" style="color:{color};font-size:1.5rem;">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)


# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr>
<p style="text-align:center; color:#334155; font-size:0.8rem;">
    🧠 BurnoutIQ · AI-Powered Academic Burnout Detection ·
    Built with Random Forest & Streamlit ·
    <span style="color:#6366f1;">Privacy-First Architecture</span>
</p>
""", unsafe_allow_html=True)

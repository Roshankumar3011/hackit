"""
Reusable Plotly visualizations for the Burnout Detection dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def gauge_chart(confidence: float, label: str, color: str) -> go.Figure:
    """Speedometer-style gauge showing confidence %."""
    value = round(confidence * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": "%", "font": {"size": 42, "color": "white"}},
        title={"text": f"<b>{label}</b>", "font": {"size": 20, "color": color}},
        delta={"reference": 50, "increasing": {"color": color}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
            "bar":       {"color": color, "thickness": 0.35},
            "bgcolor":   "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#0f172a"},
                {"range": [40, 70], "color": "#1e293b"},
                {"range": [70, 100],"color": "#0f172a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.80,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=280,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def probability_bar(probabilities: dict) -> go.Figure:
    """Stacked horizontal bars for class probabilities."""
    labels = list(probabilities.keys())
    values = [round(v * 100, 1) for v in probabilities.values()]
    colors = ["#22c55e", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    for lbl, val, col in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            name=lbl,
            x=[val],
            y=["Probability"],
            orientation="h",
            marker_color=col,
            text=f"{val}%",
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        legend=dict(orientation="h", y=-0.3, font={"size": 12}),
        height=130,
        margin=dict(t=10, b=50, l=10, r=10),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


def feature_importance_chart(importances: "pd.Series") -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    labels = importances.index.tolist()
    values = importances.values.tolist()

    FRIENDLY = {
        "login_frequency":       "Login Frequency",
        "avg_study_duration":    "Avg Study Duration",
        "assignment_delay_days": "Assignment Delay",
        "forum_participation":   "Forum Participation",
        "quiz_score_avg":        "Quiz Score Avg",
        "resource_access_count": "Resource Access Count",
        "missed_deadlines_pct":  "Missed Deadlines %",
        "session_break_freq":    "Session Break Frequency",
    }
    labels = [FRIENDLY.get(l, l) for l in labels]

    bar_colors = [
        f"rgba(99,102,241,{0.4 + 0.6 * (v / max(values))})" for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        marker_line_color="rgba(99,102,241,0.9)",
        marker_line_width=1,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#94a3b8"},
    ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        xaxis=dict(title="Importance Score", color="#64748b", gridcolor="#1e293b"),
        yaxis=dict(color="#cbd5e1"),
        height=360,
        margin=dict(t=20, b=40, l=20, r=80),
    )
    return fig


def batch_risk_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of risk distribution from batch results."""
    counts = df["burnout_risk_label"].value_counts()
    color_map = {"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
    colors = [color_map.get(l, "#94a3b8") for l in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=3)),
        textinfo="percent+label",
        textfont={"size": 13, "color": "white"},
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        legend=dict(font={"color": "#e2e8f0"}),
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        annotations=[dict(text="Risk Mix", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color="#94a3b8"))],
    )
    return fig


def batch_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of confidence scores from batch results."""
    fig = px.histogram(
        df, x="confidence", nbins=20,
        color="burnout_risk_label",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        labels={"confidence": "Confidence (%)", "burnout_risk_label": "Risk Level"},
        barmode="overlay",
        opacity=0.75,
    )
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        height=300,
        legend=dict(font={"color": "#e2e8f0"}),
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig

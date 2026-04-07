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


def confusion_matrix_chart(cm: list, labels: list) -> go.Figure:
    """
    Annotated heatmap for the confusion matrix.

    Parameters
    ----------
    cm     : 2-D list  [true_class][pred_class]
    labels : list of class name strings
    """
    cm_arr   = [[int(v) for v in row] for row in cm]
    row_sums = [max(sum(row), 1) for row in cm_arr]

    # Normalize rows for colour intensity (0-1), keep raw counts for annotations
    cm_norm = [
        [round(cm_arr[r][c] / row_sums[r], 3) for c in range(len(labels))]
        for r in range(len(labels))
    ]

    annotations = []
    for r in range(len(labels)):
        for c in range(len(labels)):
            count = cm_arr[r][c]
            pct   = cm_norm[r][c] * 100
            annotations.append(dict(
                x=labels[c],
                y=labels[r],
                text=f"<b>{count}</b><br><span style='font-size:11px'>{pct:.1f}%</span>",
                showarrow=False,
                font=dict(size=14, color="white"),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#0d1827"],
            [0.4, "#1e3a5f"],
            [0.7, "#3b4dd9"],
            [1.0, "#6366f1"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Row %", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        annotations=annotations,
        xaxis=dict(
            title="Predicted Label",
            color="#94a3b8",
            tickfont=dict(color="#cbd5e1"),
            side="bottom",
        ),
        yaxis=dict(
            title="True Label",
            color="#94a3b8",
            tickfont=dict(color="#cbd5e1"),
            autorange="reversed",
        ),
        height=380,
        margin=dict(t=30, b=60, l=100, r=40),
    )
    return fig


def per_class_metrics_chart(per_class: dict) -> go.Figure:
    """
    Grouped bar chart: Precision / Recall / F1 per class.

    Parameters
    ----------
    per_class : dict  label -> {precision, recall, f1-score, support}
    """
    labels    = list(per_class.keys())
    precision = [round(per_class[l]["precision"] * 100, 1) for l in labels]
    recall    = [round(per_class[l]["recall"]    * 100, 1) for l in labels]
    f1        = [round(per_class[l]["f1-score"]  * 100, 1) for l in labels]

    fig = go.Figure()
    for metric, values, color in [
        ("Precision", precision, "#6366f1"),
        ("Recall",    recall,    "#22c55e"),
        ("F1-Score",  f1,        "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(
            name=metric,
            x=labels,
            y=values,
            marker_color=color,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        yaxis=dict(
            range=[0, 120],
            title="Score (%)",
            gridcolor="#1e293b",
            color="#94a3b8",
        ),
        xaxis=dict(color="#cbd5e1"),
        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(color="#e2e8f0"),
        ),
        height=360,
        margin=dict(t=20, b=70, l=50, r=20),
    )
    return fig

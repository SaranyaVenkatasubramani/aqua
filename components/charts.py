from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TEXT = "rgba(255,255,255,.76)"
GRID = "rgba(255,255,255,.10)"
AQUA = "#4FC3F7"
CORAL = "#FF7A59"
MINT = "#3DDC97"


def base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=14, t=22, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat, sans-serif", color=TEXT, size=11),
        hoverlabel=dict(
            bgcolor="#060338",
            bordercolor="rgba(79,195,247,.35)",
            font=dict(color="#FFFFFF", family="Montserrat, sans-serif"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=TEXT),
            linecolor=GRID,
            title_font=dict(color=TEXT),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=TEXT),
            linecolor=GRID,
            title_font=dict(color=TEXT),
        ),
    )
    return fig


def pressure_flow_chart(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    end: int,
    window: int = 288,
    threshold: float = 0.5,
) -> go.Figure:
    start = max(0, end - window + 1)
    sample = frame.iloc[start : end + 1]
    feature_sample = features.iloc[start : end + 1]
    probs = probabilities[start : end + 1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if len(sample):
        risky = probs >= threshold
        edges = np.diff(np.r_[False, risky, False].astype(int))
        for left, right in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            fig.add_vrect(
                x0=sample.index[left],
                x1=sample.index[min(right - 1, len(sample) - 1)],
                fillcolor="rgba(255,122,89,.17)",
                line_width=0,
                layer="below",
            )
        fig.add_trace(
            go.Scatter(
                x=sample.index,
                y=feature_sample["mean_pressure"],
                name="Network pressure",
                mode="lines",
                line=dict(color=AQUA, width=2),
                hovertemplate="%{x|%b %d, %H:%M}<br>Pressure %{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sample.index,
                y=feature_sample["total_flow"],
                name="Total flow",
                mode="lines",
                line=dict(color="#BFE9FF", width=1.5, dash="dot"),
                hovertemplate="%{x|%b %d, %H:%M}<br>Flow %{y:.1f}<extra></extra>",
            ),
            secondary_y=True,
        )
    base_layout(fig, 350)
    fig.update_yaxes(title_text="Pressure", secondary_y=False, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(
        title_text="Flow",
        secondary_y=True,
        showgrid=False,
        range=[0, max(float(frame["p227"].max() + frame["p235"].max()), 1)],
    )
    return fig


def profile_history_chart(
    actual: pd.Series,
    expected: pd.Series,
    sensor: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=expected.values,
            name="Expected profile",
            mode="lines",
            line=dict(color="rgba(191,233,255,.68)", width=1.7, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            name="Observed pressure",
            mode="lines",
            line=dict(color=AQUA, width=2),
            fill="tonexty",
            fillcolor="rgba(79,195,247,.08)",
        )
    )
    fig.update_layout(yaxis_title=f"{sensor} pressure")
    return base_layout(fig, 300)


def episode_timeline(episodes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not episodes.empty:
        for i, episode in episodes.iterrows():
            color = (
                CORAL
                if episode.get("severity", "") == "High"
                else "#F6B35B"
                if episode.get("severity", "") == "Elevated"
                else AQUA
            )
            fig.add_trace(
                go.Scatter(
                    x=[episode["start"], episode["end"]],
                    y=[i, i],
                    mode="lines+markers",
                    line=dict(color=color, width=8),
                    marker=dict(size=8, color=color),
                    name=f"Episode {i + 1}",
                    showlegend=False,
                    hovertemplate=(
                        f"Episode {i + 1}<br>%{{x|%Y-%m-%d %H:%M}}"
                        f"<br>{episode['days']:.2f} days<extra></extra>"
                    ),
                )
            )
    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    return base_layout(fig, max(220, min(400, 180 + len(episodes) * 20)))


def confusion_matrix_chart(matrix: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted normal", "Predicted leak"],
            y=["Actual normal", "Actual leak"],
            text=matrix,
            texttemplate="%{text:,}",
            textfont=dict(size=20, color="#FFFFFF"),
            colorscale=[[0, "#11106b"], [0.45, "#1B3FA8"], [1, "#4FC3F7"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>%{z:,} readings<extra></extra>",
        )
    )
    base_layout(fig, 300)
    fig.update_layout(
        xaxis=dict(side="bottom", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return fig

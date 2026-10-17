from __future__ import annotations

import hashlib
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from streamlit_autorefresh import st_autorefresh

from components.charts import (
    AQUA,
    CORAL,
    MINT,
    confusion_matrix_chart,
    episode_timeline,
    pressure_flow_chart,
    profile_history_chart,
)
from components.engine import (
    LABEL_COL,
    MODEL_PATH,
    ROOT,
    build_features,
    extract_episodes,
    find_dataset,
    load_dataset,
    load_dataset_bytes,
    load_or_train_bundle,
)


st.set_page_config(
    page_title="AquaSentinel · Water intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGES = ["Home", "Network", "Detect", "Episodes", "About"]
PRESSURE_TARGET = 0.5


def install_styles() -> None:
    css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    bubbles = "".join(
        f'<i style="--left:{(i * 37 + 11) % 100}%;--size:{6 + (i * 7) % 13}px;'
        f'--duration:{22 + (i * 11) % 30}s;--delay:-{(i * 13) % 42}s"></i>'
        for i in range(27)
    )
    st.markdown(
        f'<div class="ocean-bubbles" aria-hidden="true">{bubbles}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_runtime(
    csv_path: str,
    csv_modified: float,
    model_path: str,
    model_modified: float,
    uploaded_csv: bytes | None,
    upload_digest: str,
) -> dict:
    del csv_modified, model_modified, upload_digest
    frame = (
        load_dataset_bytes(uploaded_csv, "uploaded CSV")
        if uploaded_csv is not None
        else load_dataset(csv_path)
    )
    model_file = Path(model_path)
    training_frame = frame
    if uploaded_csv is not None and not model_file.is_file():
        training_frame = load_dataset(csv_path)
    bundle = load_or_train_bundle(training_frame, model_path)
    pressure_cols = list(bundle["profiles"]["pressure_cols"])
    expected_features = bundle["feature_cols"]
    features = build_features(frame, bundle["profiles"])
    missing = [name for name in expected_features if name not in features.columns]
    if missing:
        raise ValueError(
            "This dataset is not compatible with the supplied model. "
            f"Missing engineered features: {', '.join(missing[:8])}"
        )
    features = features.loc[:, expected_features]
    probabilities = bundle["model"].predict_proba(
        features.values.astype(np.float32)
    )[:, 1]
    residual_cols = [f"resid_{sensor}" for sensor in pressure_cols]
    residual_std = bundle.get("residual_std")
    if residual_std is None:
        clean_2018 = (frame.index.year == 2018) & (frame[LABEL_COL].values == 0)
        if clean_2018.any():
            residual_std = features.loc[clean_2018, residual_cols].std()
        else:
            training_frame = load_dataset(csv_path)
            training_features = build_features(training_frame, bundle["profiles"])
            clean_2018 = (
                (training_frame.index.year == 2018)
                & (training_frame[LABEL_COL].values == 0)
            )
            residual_std = training_features.loc[clean_2018, residual_cols].std()
        residual_std = residual_std.clip(lower=bundle["profiles"]["std_floor"])
    return {
        "frame": frame,
        "features": features,
        "probabilities": probabilities,
        "bundle": bundle,
        "pressure_cols": pressure_cols,
        "residual_std": residual_std,
        "csv_path": csv_path,
        "uploaded": uploaded_csv is not None,
    }


def section_heading(title: str, subtitle: str, label: str = "AQUASENTINEL") -> None:
    st.markdown(
        f'<div class="section-heading"><div><div class="eyebrow">{html.escape(label)}</div>'
        f'<h1>{html.escape(title)}</h1></div>'
        f'<p>{html.escape(subtitle)}</p></div><div class="section-rule"></div>',
        unsafe_allow_html=True,
    )


def render_header() -> str:
    if st.session_state.get("pending_page"):
        st.session_state.nav = st.session_state.pop("pending_page")
    left, middle, right = st.columns([2.15, 7.2, 1.15], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="brand" aria-label="AquaSentinel"><span>Aqua</span>'
            '<strong>Sentinel</strong></div>',
            unsafe_allow_html=True,
        )
    with middle:
        page = st.radio(
            "Main navigation",
            PAGES,
            horizontal=True,
            label_visibility="collapsed",
            key="nav",
        )
    with right:
        if st.button("▦", help="Open dashboard overview", key="grid_nav"):
            st.session_state.pending_page = "About"
            st.rerun()
        st.markdown(
            '<div class="topbar-status"><span class="status-dot"></span>NETWORK LIVE</div>',
            unsafe_allow_html=True,
        )
    return page


def score_sensors(runtime: dict, end: int, window: int = 72) -> pd.Series:
    pressure_cols = runtime["pressure_cols"]
    resid_cols = [f"resid_{sensor}" for sensor in pressure_cols]
    start = max(0, end - window + 1)
    residual = runtime["features"].iloc[start : end + 1][resid_cols].mean()
    peer_adjusted = residual - residual.median()
    scores = peer_adjusted / runtime["residual_std"]
    scores.index = pressure_cols
    return scores.replace([np.inf, -np.inf], np.nan).fillna(0)


def expected_profile(
    timestamps: pd.DatetimeIndex,
    sensor: str,
    profiles: dict,
) -> pd.Series:
    slots = timestamps.hour * 12 + timestamps.minute // 5
    daytypes = (timestamps.dayofweek >= 5).astype(int)
    values = np.array(
        [
            profiles["pressure"][int(daytype)][int(slot)][
                profiles["pressure_cols"].index(sensor)
            ]
            for slot, daytype in zip(slots, daytypes)
        ]
    )
    return pd.Series(values, index=timestamps)


def render_home(runtime: dict) -> None:
    frame = runtime["frame"]
    last_active = int(frame[LABEL_COL].iloc[-1] == 1)
    sensors_online = sum(
        pd.notna(frame[sensor].iloc[-1]) for sensor in runtime["pressure_cols"]
    )
    metrics = calculate_metrics(runtime)
    recall = metrics["recall"] * 100 if metrics["recall"] is not None else "N/A"
    recall_note = (
        "2019 temporal holdout · 0.50 cutoff"
        if metrics["recall"] is not None
        else "No labeled 2019 holdout in this feed"
    )
    cards = [
        ("ACTIVE LEAKS", last_active, "At the latest observed reading"),
        ("SENSORS ONLINE", sensors_online, "Pressure channels reporting"),
        ("WATER SAVED (L)", 0, "No repair/intervention ledger connected"),
        ("MODEL RECALL", recall, recall_note),
    ]
    st.markdown(
        '<section class="hero" aria-label="AquaSentinel introduction">'
        '<div class="hero-copy"><div class="eyebrow">PRESENTATION · WATER INTELLIGENCE</div>'
        '<h1 class="hero-title">Leak<br>Detection</h1>'
        '<div class="hero-rule"></div>'
        '<p class="hero-tagline">From Our Taps to the Deep Blue</p></div>'
        '<div class="hero-image-wrap"><img class="hero-image" '
        'src="https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=1600&q=85" '
        'alt="Blue underwater seascape illuminated by sunbeams"></div>'
        '<div class="hero-splash" aria-hidden="true"><svg viewBox="0 0 180 120" fill="none">'
        '<path d="M87 110c-9-22-29-26-23-45 3-10 10-13 11-25 1-11-6-20-2-30 7 11 9 18 13 25 '
        '4-23 13-32 21-35-5 19 1 29 7 39 7-18 17-24 28-27-9 19-11 31-8 45 7-8 16-12 27-13 '
        '-16 15-11 37-28 50-11 8-27 10-46 16Z" fill="currentColor" opacity=".7"/>'
        '<circle cx="45" cy="32" r="5" fill="currentColor"/><circle cx="143" cy="22" r="3" fill="currentColor"/>'
        '</svg></div>'
        '<div class="hero-wave" aria-hidden="true"><svg viewBox="0 0 240 90" fill="none">'
        '<path d="M0 58c32-30 52 27 83 1s50-42 78-15 47 33 79 1M0 76c31-27 50 23 81 1s50-39 78-13 '
        '47 30 81 0" stroke="currentColor" stroke-width="5" stroke-linecap="round"/>'
        '<path d="M0 88c30-19 49 18 80 1s51-30 79-9 49 22 81-1" stroke="currentColor" stroke-width="2"/>'
        '</svg></div></section>',
        unsafe_allow_html=True,
    )
    live_col, upload_col, _ = st.columns([1.35, 1.45, 6.4])
    with live_col:
        if st.button("◉  Live Monitor", key="hero_monitor"):
            st.session_state.pending_page = "Detect"
            st.rerun()
    with upload_col:
        if st.button("↑  Upload Data", key="hero_upload"):
            st.session_state.show_uploader = not st.session_state.get(
                "show_uploader", False
            )
    if st.session_state.get("show_uploader"):
        uploaded = st.file_uploader(
            "Choose a compatible 5-minute sensor CSV",
            type=["csv"],
            help="Required columns match the supplied AquaSentinel sensor dataset.",
            key="dataset_upload",
        )
        if uploaded is not None:
            content = uploaded.getvalue()
            st.session_state.uploaded_csv = content
            st.session_state.upload_digest = hashlib.sha256(content).hexdigest()
            st.success(
                f"{uploaded.name} loaded for scoring with the current trained model."
            )
            if st.button("Apply uploaded data", key="apply_upload"):
                st.rerun()

    card_items = []
    for label, value, note in cards:
        static = isinstance(value, str)
        target = 0 if static else value
        decimals = 1 if label == "MODEL RECALL" and not static else 0
        display = value if static else f"{value:.{decimals}f}"
        card_items.append(
            '<div class="glass-card kpi-card"><div class="kpi-label">'
            f"{label}</div><div class=\"kpi-value\" data-target=\"{target}\" "
            f'data-decimals="{decimals}" data-static="{str(static).lower()}">'
            f"{display}</div><div class=\"kpi-note\">{note}</div></div>"
        )
    card_markup = "".join(card_items)
    components.html(
        f"""<div class="kpi-grid">{card_markup}</div>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Space+Grotesk:wght@600&display=swap');
        *{{box-sizing:border-box}}body{{margin:0;background:transparent;font-family:Montserrat,Arial,sans-serif;color:white}}
        .kpi-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;padding:4px 1px 8px}}
        .glass-card{{border:1px solid rgba(255,255,255,.16);border-radius:20px;background:rgba(255,255,255,.055);
        padding:18px 19px;min-height:124px;backdrop-filter:blur(14px);box-shadow:0 12px 35px rgba(3,2,35,.13)}}
        .kpi-label{{color:#BFE9FF;font-size:10px;font-weight:700;letter-spacing:1.6px}}
        .kpi-value{{margin:10px 0 3px;color:#fff;font:600 29px/1.1 'Space Grotesk',monospace;font-variant-numeric:tabular-nums}}
        .kpi-note{{color:rgba(255,255,255,.72);font-size:10px;line-height:1.45}}
        .glass-card:hover{{border-color:rgba(79,195,247,.52);box-shadow:0 12px 36px rgba(79,195,247,.12);transform:translateY(-2px)}}
        @media(max-width:800px){{.kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}}.glass-card{{min-height:112px;padding:14px}}}}
        @media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
        </style>
        <script>
        const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        document.querySelectorAll('.kpi-value').forEach(el=>{{
          const target=Number(el.dataset.target), decimals=Number(el.dataset.decimals||0);
          if(el.dataset.static==='true')return;
          if(reduced){{el.textContent=target.toLocaleString(undefined,{{minimumFractionDigits:decimals,maximumFractionDigits:decimals}});return;}}
          const start=performance.now(), duration=1050;
          function tick(now){{const t=Math.min((now-start)/duration,1), eased=1-Math.pow(1-t,3);
            el.textContent=(target*eased).toLocaleString(undefined,{{minimumFractionDigits:decimals,maximumFractionDigits:decimals}});
            if(t<1)requestAnimationFrame(tick);
          }}requestAnimationFrame(tick);
        }});
        </script>""",
        height=160,
        scrolling=False,
    )
    st.markdown(
        '<div class="wave-divider" aria-hidden="true"><svg viewBox="0 0 1200 60" '
        'preserveAspectRatio="none"><path d="M0 24 C130 6 190 46 320 25 S510 8 620 26 '
        's190 22 300 0 180-15 280 4" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<path d="M0 37 C120 20 210 55 330 36 S520 18 650 38 840 56 960 34 '
        's165-13 240 2" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        '</svg></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">BELOW THE SURFACE</div>'
        '<h2>One network. Earlier answers.</h2></div>'
        '<p>Operational signals become actionable context.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-rule"></div><div class="soft-panel">'
        '<div class="eyebrow">FROM SENSOR TO SIGNAL</div>'
        '<p class="muted">AquaSentinel learns the clean daily rhythm of the network, '
        'then watches for pressure and flow behavior that does not belong. Every '
        'feature is causal: the model only sees readings available up to that moment.</p>'
        f'<span class="pill">{len(runtime["pressure_cols"])} pressure sensors</span> '
        '<span class="pill">5-minute cadence</span> '
        '<span class="pill">2018 → 2019 temporal validation</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing {len(frame):,} readings · {frame.index.min():%b %Y}–"
        f"{frame.index.max():%b %Y} · {runtime['pressure_cols'].__len__()} pressure nodes"
        + (" · uploaded feed" if runtime["uploaded"] else "")
    )


def network_svg(runtime: dict, scores: pd.Series) -> str:
    cols = runtime["pressure_cols"]
    coords = {}
    for idx, sensor in enumerate(cols):
        row, col = divmod(idx, 7)
        coords[sensor] = (75 + col * 120 + (row % 2) * 25, 58 + row * 88)
    links = []
    for idx, sensor in enumerate(cols):
        row, col = divmod(idx, 7)
        if col < 6 and idx + 1 < len(cols):
            links.append((sensor, cols[idx + 1]))
        if row < 4 and idx + 7 < len(cols):
            links.append((sensor, cols[idx + 7]))
        if row < 4 and col > 0 and idx + 6 < len(cols):
            links.append((sensor, cols[idx + 6]))
    paths = "".join(
        f'<path d="M{x1},{y1} L{x2},{y2}" stroke="rgba(79,195,247,.22)" '
        'stroke-width="2" stroke-dasharray="3 5"/>'
        for first, second in links
        for x1, y1 in [coords[first]]
        for x2, y2 in [coords[second]]
    )
    nodes = []
    for sensor, (x, y) in coords.items():
        score = float(scores.get(sensor, 0))
        anomaly = abs(score) >= 2
        color = CORAL if anomaly else MINT
        pulse = " class='pulse'" if anomaly else ""
        nodes.append(
            f'<g tabindex="0" role="img" aria-label="{html.escape(sensor)}: '
            f'{"anomaly" if anomaly else "normal"}, z score {score:.1f}">'
            f'<title>{html.escape(sensor)} · peer z-score {score:+.2f}</title>'
            f'<circle cx="{x}" cy="{y}" r="13" fill="{color}" opacity=".17"/>'
            f'<circle cx="{x}" cy="{y}" r="5.7" fill="{color}"{pulse}/>'
            f'<text x="{x}" y="{y + 24}" text-anchor="middle" fill="#BFE9FF" '
            f'font-size="10" font-family="Space Grotesk,monospace">{html.escape(sensor)}</text>'
            '</g>'
        )
    return (
        '<div class="soft-panel network-map"><svg viewBox="0 0 900 440" role="img" '
        'aria-label="Schematic water distribution network; green nodes are healthy, '
        'coral nodes are anomalous">'
        f"<g>{paths}</g>{''.join(nodes)}</svg></div>"
    )


def render_network(runtime: dict) -> None:
    frame = runtime["frame"]
    end = len(frame) - 1
    scores = score_sensors(runtime, end)
    section_heading(
        "Network overview",
        f"{len(runtime['pressure_cols'])} pressure nodes · schematic connections · latest observation",
        "DISTRIBUTION MAP",
    )
    st.markdown(
        '<style>@keyframes node-pulse{0%,100%{filter:drop-shadow(0 0 3px #FF7A59);opacity:1}'
        '50%{filter:drop-shadow(0 0 11px #FF7A59);opacity:.55}}'
        '.pulse{animation:node-pulse 1.7s ease-in-out infinite;transform-box:fill-box;transform-origin:center}'
        '.network-map svg{display:block;width:100%;height:auto;max-height:470px}'
        '@media(prefers-reduced-motion:reduce){.pulse{animation:none}}</style>',
        unsafe_allow_html=True,
    )
    map_col, detail_col = st.columns([1.72, 1], gap="large")
    with map_col:
        st.markdown(network_svg(runtime, scores), unsafe_allow_html=True)
        st.markdown(
            f'<span class="pill"><span class="healthy">●</span> Normal</span> '
            f'<span class="pill"><span class="alert">●</span> Deviation ≥ 2σ</span> '
            f'<span class="pill">As of {frame.index[-1]:%d %b %Y %H:%M}</span>',
            unsafe_allow_html=True,
        )
    with detail_col:
        sensor = st.session_state.get("selected_sensor", runtime["pressure_cols"][0])
        if sensor not in runtime["pressure_cols"]:
            sensor = runtime["pressure_cols"][0]
        score = float(scores.get(sensor, 0))
        color = "alert" if abs(score) >= 2 else "healthy"
        st.markdown(
            f'<div class="eyebrow">SENSOR DETAIL</div><h2 class="mono">{sensor}</h2>'
            f'<p class="{color}">● {"Anomalous" if abs(score) >= 2 else "Within expected range"}'
            f' &nbsp; <span class="muted">peer z-score {score:+.2f}</span></p>',
            unsafe_allow_html=True,
        )
        start = max(0, end - 287)
        actual = frame[sensor].iloc[start : end + 1]
        expected = expected_profile(
            actual.index, sensor, runtime["bundle"]["profiles"]
        )
        st.plotly_chart(
            profile_history_chart(actual, expected, sensor),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown(
            '<div class="eyebrow">SELECT A NODE</div>',
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(runtime["pressure_cols"]), 7):
            sensors = runtime["pressure_cols"][row_start : row_start + 7]
            buttons = st.columns(len(sensors))
            for column, node in zip(buttons, sensors):
                with column:
                    marker = "🔴" if abs(scores.get(node, 0)) >= 2 else "🟢"
                    if st.button(
                        f"{marker} {node}",
                        key=f"select_{node}",
                        help=f"Show the pressure profile for {node}",
                    ):
                        st.session_state.selected_sensor = node
                        st.rerun()


def render_gauge(probability: float, threshold: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#FFFFFF"}},
            title={"text": "LEAK PROBABILITY", "font": {"size": 11, "color": "#BFE9FF"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "rgba(255,255,255,.5)",
                    "tickfont": {"color": "#BFE9FF"},
                },
                "bar": {"color": CORAL if probability >= threshold else AQUA, "thickness": .24},
                "bgcolor": "rgba(255,255,255,.04)",
                "bordercolor": "rgba(255,255,255,.12)",
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(79,195,247,.10)"},
                    {"range": [threshold * 100, 100], "color": "rgba(255,122,89,.12)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": .8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=18, r=18, t=48, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat,sans-serif", color="#FFFFFF"),
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_detect(runtime: dict) -> None:
    frame = runtime["frame"]
    features = runtime["features"]
    probabilities = runtime["probabilities"]
    threshold = runtime["bundle"]["thresholds"]["sensitive"]
    if "detect_index" not in st.session_state:
        default = int(np.searchsorted(frame.index, pd.Timestamp("2019-01-01")))
        st.session_state.detect_index = min(default, len(frame) - 1)
    if st.session_state.get("sim_playing", False):
        st.session_state.detect_index = min(
            int(st.session_state.detect_index) + 12, len(frame) - 1
        )
        if st.session_state.detect_index >= len(frame) - 1:
            st.session_state.sim_playing = False

    section_heading(
        "Live simulation",
        "Replay the sensor stream · one tick advances one simulated hour",
        "CAUSAL INFERENCE",
    )
    control, timestamp_col = st.columns([1, 4], vertical_alignment="center")
    with control:
        label = "Ⅱ  Pause" if st.session_state.get("sim_playing", False) else "▶  Play"
        if st.button(label, key="simulation_toggle"):
            st.session_state.sim_playing = not st.session_state.get(
                "sim_playing", False
            )
            st.rerun()
    with timestamp_col:
        current = st.slider(
            "Observation index",
            min_value=0,
            max_value=len(frame) - 1,
            step=1,
            key="detect_index",
            label_visibility="collapsed",
            help="Move through the historical 5-minute sensor stream.",
        )
    if st.session_state.get("sim_playing", False):
        st_autorefresh(interval=1000, limit=None, key="aqua_simulation_tick")
    timestamp = frame.index[current]
    probability = float(probabilities[current])
    current_scores = score_sensors(runtime, current)
    suspects = current_scores.reindex(
        current_scores.abs().sort_values(ascending=False).index
    ).head(5)
    st.markdown(
        f'<span class="pill">● STREAMING</span> '
        f'<span class="pill mono">{timestamp:%A, %d %B %Y · %H:%M}</span> '
        f'<span class="pill">5-minute interval</span>',
        unsafe_allow_html=True,
    )
    chart_col, gauge_col = st.columns([1.75, 1], gap="large")
    with chart_col:
        st.markdown('<div class="eyebrow">PRESSURE + FLOW · LAST 24 HOURS</div>', unsafe_allow_html=True)
        st.plotly_chart(
            pressure_flow_chart(
                frame,
                features,
                probabilities,
                current,
                threshold=threshold,
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with gauge_col:
        render_gauge(probability, threshold)
        state = "LEAK LIKELY" if probability >= threshold else "NETWORK NORMAL"
        color = "alert" if probability >= threshold else "healthy"
        st.markdown(
            f'<div class="soft-panel"><div class="eyebrow">CURRENT ALERT</div>'
            f'<h2 class="{color}">{state}</h2>'
            f'<p class="muted">Decision threshold {threshold:.0%} · '
            f'causal model score {probability:.1%}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="eyebrow">TOP SUSPECT SENSORS · PEER-NORMALIZED DEVIATION</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(len(suspects))
    for column, (sensor, score) in zip(columns, suspects.items()):
        with column:
            tone = "alert" if abs(score) >= 2 else "healthy"
            st.markdown(
                f'<div class="glass-card"><div class="eyebrow">{html.escape(sensor)}</div>'
                f'<h2 class="mono {tone}">{score:+.2f}σ</h2>'
                f'<div class="muted">{"Investigate" if abs(score) >= 2 else "In range"}</div></div>',
                unsafe_allow_html=True,
            )
    if st.session_state.get("sim_playing", False):
        st.caption("Simulation advances 12 readings (one hour) every second.")


def enrich_episodes(runtime: dict) -> pd.DataFrame:
    frame = runtime["frame"]
    episodes = extract_episodes(frame[LABEL_COL])
    if episodes.empty:
        return episodes.assign(severity=pd.Series(dtype=str), probability=pd.Series(dtype=float))
    severity = []
    probabilities = []
    scores = runtime["probabilities"]
    for row in episodes.itertuples(index=False):
        start = int(frame.index.searchsorted(row.start, side="left"))
        end = int(frame.index.searchsorted(row.end, side="right"))
        avg = float(np.mean(scores[start:end])) if end > start else 0.0
        probabilities.append(avg)
        severity.append("High" if avg >= 0.8 else "Elevated" if avg >= 0.5 else "Watch")
    episodes["probability"] = probabilities
    episodes["severity"] = severity
    return episodes


def render_episodes(runtime: dict) -> None:
    section_heading(
        "Leak episodes",
        "Observed label runs, merged across gaps shorter than one hour",
        "INCIDENT HISTORY",
    )
    episodes = enrich_episodes(runtime)
    if episodes.empty:
        st.info(
            "No labeled leak episodes were found in this feed. "
            "The classifier is still scoring every sensor reading."
        )
        return
    st.markdown(
        f'<span class="pill">{len(episodes)} merged episodes</span> '
        f'<span class="pill">Severity based on mean model probability</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow" style="margin-top:1.5rem">EPISODE TIMELINE</div>', unsafe_allow_html=True)
    st.plotly_chart(
        episode_timeline(episodes),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    display = episodes.copy()
    display.insert(0, "episode", [f"EP-{i:03d}" for i in range(1, len(display) + 1)])
    display["duration_days"] = display.pop("days").round(3)
    display["model_probability"] = (display.pop("probability") * 100).round(1)
    display["start"] = display["start"].dt.strftime("%Y-%m-%d %H:%M")
    display["end"] = display["end"].dt.strftime("%Y-%m-%d %H:%M")
    display = display[
        [
            "episode",
            "start",
            "end",
            "duration_days",
            "severity",
            "model_probability",
        ]
    ]
    display.columns = [
        "Episode",
        "Start",
        "End",
        "Duration (days)",
        "Severity",
        "Mean model probability (%)",
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    csv_data = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓  Export episodes as CSV",
        data=csv_data,
        file_name="aquasentinel_leak_episodes.csv",
        mime="text/csv",
        key="export_episodes",
    )
    st.caption(
        "Severity is an operational ranking derived from the model score; "
        "the episode start/end boundaries come from Leak_Status labels."
    )


def calculate_metrics(runtime: dict) -> dict[str, float | int | np.ndarray | None]:
    frame = runtime["frame"]
    test_mask = frame.index.year == 2019
    y_true = frame.loc[test_mask, LABEL_COL].astype(int).values
    probabilities = runtime["probabilities"][test_mask]
    if not len(y_true):
        return {
            "recall": None,
            "precision": None,
            "f1": None,
            "accuracy": None,
            "auc": None,
            "confusion": np.zeros((2, 2), dtype=int),
            "count": 0,
        }
    y_pred = probabilities >= PRESSURE_TARGET
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": (
            roc_auc_score(y_true, probabilities)
            if np.unique(y_true).size == 2
            else None
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "count": int(test_mask.sum()),
    }


def render_about(runtime: dict) -> None:
    section_heading(
        "Built for the unseen",
        "A transparent, time-aware pipeline for municipal water networks",
        "THE AQUASENTINEL METHOD",
    )
    step_data = [
        ("01", "Clean profiles", "Weekday/weekend medians · 288 daily slots · 2018 leak-free rows only"),
        ("02", "Pressure residuals", "Observed minus expected; diagnostic sensor z-scores use a 0.10 standard-deviation floor"),
        ("03", "Causal windows", "Backward-only 1h, 6h and 24h rolling summaries; no future readings"),
        ("04", "Classifier", "Balanced HistGradientBoosting · 150 iterations · fitted on 2018"),
    ]
    cols = st.columns(4, gap="small")
    for column, (number, title, text) in zip(cols, step_data):
        with column:
            st.markdown(
                f'<div class="glass-card" style="height:100%"><div class="eyebrow">{number}</div>'
                f'<h3>{title}</h3><p class="muted">{text}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">VALIDATION DESIGN</div>'
        '<h2>Time moves forward.</h2></div>'
        '<p>No random split. No peek into tomorrow.</p></div>'
        '<div class="section-rule"></div>'
        '<div class="soft-panel" style="display:grid;grid-template-columns:1fr auto 1fr;'
        'gap:1rem;align-items:center">'
        '<div><span class="pill">TRAIN · 2018</span><h3>Profiles + model fit</h3>'
        '<p class="muted">Profiles use only clean training rows. Classifier learns from 2018 labels.</p></div>'
        '<div class="mono" style="font-size:2rem;color:#4FC3F7" aria-hidden="true">→</div>'
        '<div><span class="pill">TEST · 2019</span><h3>Held-out evaluation</h3>'
        '<p class="muted">Every reported score below is computed on the untouched 2019 period.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    metrics = calculate_metrics(runtime)
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">MODEL REPORT CARD</div>'
        '<h2>2019 holdout</h2></div>'
        '<p>Single operating point · threshold 0.50 · raw classifier output</p></div>',
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(5)
    for column, name, value in zip(
        metric_cols,
        ["Recall", "Precision", "F1 score", "Accuracy", "ROC AUC"],
        [
            metrics["recall"],
            metrics["precision"],
            metrics["f1"],
            metrics["accuracy"],
            metrics["auc"],
        ],
    ):
        with column:
            st.metric(name, f"{value:.1%}" if value is not None else "N/A")
    matrix_col, facts_col = st.columns([1.05, 1], gap="large")
    with matrix_col:
        st.markdown('<div class="eyebrow">CONFUSION MATRIX · 2019</div>', unsafe_allow_html=True)
        st.plotly_chart(
            confusion_matrix_chart(metrics["confusion"]),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with facts_col:
        st.markdown('<div class="eyebrow">RUN DETAILS</div>', unsafe_allow_html=True)
        detail = pd.DataFrame(
            {
                "Item": [
                    "Temporal split",
                    "Test readings",
                    "Pressure nodes",
                    "Profile slots",
                    "Std floor",
                    "Feature count",
                    "Threshold",
                    "Estimator",
                ],
                "Value": [
                    "Train 2018 → test 2019",
                    f"{metrics['count']:,}" if metrics["count"] else "No 2019 rows",
                    str(len(runtime["pressure_cols"])),
                    "288 per day type",
                    "0.10",
                    str(len(runtime["bundle"]["feature_cols"])),
                    "0.50",
                    "HistGradientBoosting",
                ],
            }
        )
        st.dataframe(detail, hide_index=True, width="stretch")
        st.caption(
            ("No 2019 readings are present in this feed, so holdout metrics are unavailable. "
             if metrics["count"] == 0 else "")
            + "Sensor z-scores are used for peer-normalized suspect ranking. "
            "The classifier feature set follows the notebook exactly: profile "
            "residuals, scalar signals, time features, and causal rolling summaries."
        )


def main() -> None:
    install_styles()
    page = render_header()
    try:
        csv_path = find_dataset()
        csv_modified = csv_path.stat().st_mtime
        model_modified = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
        uploaded_csv = st.session_state.get("uploaded_csv")
        digest = st.session_state.get("upload_digest", "")
        skeleton = st.empty()
        skeleton.markdown(
            '<div class="skeleton-grid" aria-label="Loading AquaSentinel metrics" '
            'aria-busy="true">'
            '<div class="skeleton-card"></div><div class="skeleton-card"></div>'
            '<div class="skeleton-card"></div><div class="skeleton-card"></div></div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Reading the network stream and warming the model…"):
            runtime = get_runtime(
                str(csv_path),
                csv_modified,
                str(MODEL_PATH),
                model_modified,
                uploaded_csv,
                digest,
            )
        skeleton.empty()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if "skeleton" in locals():
            skeleton.empty()
        st.error(f"AquaSentinel could not prepare the dashboard: {error}")
        st.info(
            "Check that the dataset is in the data folder and that the model "
            "bundle matches its sensor columns. Rebuild with `python prepare_model.py`."
        )
        st.stop()

    if page == "Home":
        render_home(runtime)
    elif page == "Network":
        render_network(runtime)
    elif page == "Detect":
        render_detect(runtime)
    elif page == "Episodes":
        render_episodes(runtime)
    else:
        render_about(runtime)

    st.markdown(
        '<div style="margin-top:3rem;padding:1rem 0;border-top:1px solid rgba(255,255,255,.12);'
        'color:rgba(255,255,255,.62);font-size:.72rem;display:flex;justify-content:space-between">'
        '<span>AquaSentinel · Water intelligence for a more resilient blue planet</span>'
        f'<span>{len(runtime["pressure_cols"])} pressure nodes · 5-minute cadence</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

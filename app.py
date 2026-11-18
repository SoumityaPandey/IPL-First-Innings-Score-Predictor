import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="IPL Score Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ipl_project_output")
MODEL_PATH = os.path.join(OUT, "linear_regression_model.joblib")
SCALER_PATH = os.path.join(OUT, "standard_scaler.joblib")
PROCESSED_PATH = os.path.join(OUT, "IPL_ML_Processed.csv")
RESULTS_PATH = os.path.join(OUT, "results.json")

FEATURES = [
    "powerplay_runs",
    "powerplay_fours",
    "powerplay_sixes",
    "powerplay_dot_balls",
    "powerplay_wickets",
    "powerplay_extras",
]

FEATURE_LABELS = {
    "powerplay_runs": "Runs",
    "powerplay_fours": "Fours",
    "powerplay_sixes": "Sixes",
    "powerplay_dot_balls": "Dot balls",
    "powerplay_wickets": "Wickets",
    "powerplay_extras": "Extras",
}

FEATURE_DESCRIPTIONS = {
    "powerplay_runs": "Runs scored in the first six overs",
    "powerplay_fours": "Fours scored in the Powerplay",
    "powerplay_sixes": "Sixes scored in the Powerplay",
    "powerplay_dot_balls": "Dot balls in the Powerplay",
    "powerplay_wickets": "Wickets lost in the Powerplay",
    "powerplay_extras": "Extras conceded during the Powerplay",
}

# Futuristic visual system
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080b12;
        --panel: rgba(16, 21, 32, 0.78);
        --panel2: rgba(20, 27, 41, 0.72);
        --line: rgba(148, 163, 184, 0.15);
        --text: #f8fafc;
        --muted: #94a3b8;
        --cyan: #38bdf8;
        --violet: #8b5cf6;
        --green: #34d399;
        --orange: #fb923c;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(56,189,248,.12), transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(139,92,246,.12), transparent 26%),
            linear-gradient(145deg, #070a10 0%, #0b1019 55%, #080b12 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,11,18,.98), rgba(13,18,29,.98));
        border-right: 1px solid rgba(148,163,184,.12);
    }
    [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 30px;
        border: 1px solid rgba(56,189,248,.18);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15,23,42,.9), rgba(17,24,39,.58));
        box-shadow: 0 24px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
        margin-bottom: 22px;
    }
    .hero:after {
        content: '';
        position: absolute;
        width: 320px; height: 320px;
        right: -120px; top: -180px;
        background: radial-gradient(circle, rgba(56,189,248,.20), transparent 65%);
        pointer-events: none;
    }
    .eyebrow { color: #67e8f9; font-size: .76rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
    .hero-title { font-family: 'Space Grotesk'; font-size: clamp(2rem, 4vw, 3.5rem); font-weight: 800; line-height: 1.02; margin: 7px 0 10px; }
    .hero-sub { color: #a7b4c8; max-width: 780px; font-size: .98rem; line-height: 1.65; }
    .pill { display:inline-block; padding: 5px 10px; border-radius: 999px; background: rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.18); color:#bae6fd; margin-right:6px; font-size:.72rem; }

    .kpi {
        background: linear-gradient(145deg, rgba(20,27,41,.86), rgba(12,17,27,.88));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 17px 18px;
        min-height: 112px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
    }
    .kpi-label { color:#94a3b8; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
    .kpi-value { font-family:'Space Grotesk'; font-size:1.75rem; font-weight:700; margin-top:8px; }
    .kpi-note { color:#64748b; font-size:.72rem; margin-top:3px; }

    .section-label { color:#cbd5e1; font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin: 10px 0 8px; }
    .insight {
        border-left: 3px solid var(--cyan);
        background: rgba(15,23,42,.55);
        border-radius: 0 14px 14px 0;
        padding: 13px 15px;
        color: #cbd5e1;
        line-height:1.55;
        margin: 8px 0;
    }
    .score-card {
        background: radial-gradient(circle at 50% 20%, rgba(56,189,248,.13), transparent 48%), rgba(10,15,24,.82);
        border:1px solid rgba(56,189,248,.18);
        border-radius:22px;
        padding: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,.22);
    }
    .mini-card { background:rgba(15,23,42,.62); border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
    .mini-title { color:#94a3b8; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
    .mini-value { font-size:1.1rem; font-weight:700; margin-top:4px; }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.55);
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 15px;
        padding: 10px 12px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(56,189,248,.22);
        background: rgba(15,23,42,.72);
        color: #e2e8f0;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: rgba(56,189,248,.55); color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def required_files_exist():
    return all(os.path.exists(p) for p in [PROCESSED_PATH, MODEL_PATH, SCALER_PATH, RESULTS_PATH])


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(PROCESSED_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


if not required_files_exist():
    st.error("Required project artifacts are missing from ipl_project_output/.")
    st.stop()

with open(RESULTS_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)

df = load_data()
model, scaler = load_model()

# Header
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">IPL • MACHINE LEARNING • 2008–2025</div>
      <div class="hero-title">First-Innings Score Intelligence</div>
      <div class="hero-sub">A futuristic analytics cockpit for exploring Powerplay patterns, evaluating the Linear Regression model, and forecasting an IPL first-innings total.</div>
      <div style="margin-top:14px"><span class="pill">Linear Regression</span><span class="pill">6 Powerplay Features</span><span class="pill">1,169 Matches</span><span class="pill">Interactive Analytics</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🏏 IPL COMMAND CENTER")
    st.caption("Explore the model from four different angles.")
    page = st.radio(
        "Navigation",
        ["Command Dashboard", "Score Predictor", "EDA Explorer", "Model Evaluation", "Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model**")
    st.caption("Linear Regression")
    st.markdown("**Target**")
    st.caption("Final first-innings score")
    st.markdown("**Input window**")
    st.caption("Powerplay • first 6 overs")

# Shared metrics
matches = int(df["match_id"].nunique())
mean_score = float(df["final_score"].mean())
median_score = float(df["final_score"].median())
rmse = float(results["rmse"])
r2 = float(results["r2"])

if page == "Command Dashboard":
    st.markdown('<div class="section-label">System overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Matches analysed</div><div class="kpi-value">{matches:,}</div><div class="kpi-note">processed IPL matches</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Average score</div><div class="kpi-value">{mean_score:.0f}</div><div class="kpi-note">runs per first innings</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Model R²</div><div class="kpi-value">{r2:.3f}</div><div class="kpi-note">explained variation</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">RMSE</div><div class="kpi-value">{rmse:.1f}</div><div class="kpi-note">typical error in runs</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="section-label">Powerplay signal</div>', unsafe_allow_html=True)
        fig = px.scatter(
            df,
            x="powerplay_runs",
            y="final_score",
            color="powerplay_wickets",
            color_continuous_scale="Turbo",
            opacity=.65,
            hover_data=["season", "batting_team", "bowling_team", "venue"],
            labels={"powerplay_runs":"Powerplay runs", "final_score":"Final score", "powerplay_wickets":"Wickets"},
        )
        fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10,r=10,t=35,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-label">Score distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="final_score", nbins=28, marginal="box", labels={"final_score":"Final score"})
        fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10,r=10,t=35,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f'<div class="insight"><b>Typical innings</b><br>Median first-innings score is <b>{median_score:.0f}</b> runs.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="insight"><b>Prediction error</b><br>The model misses by about <b>{rmse:.1f}</b> runs on average in RMSE terms.</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div class="insight"><b>Model scope</b><br>The predictor uses only the six Powerplay features shown below.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Feature cockpit</div>', unsafe_allow_html=True)
    feature_table = pd.DataFrame({"Feature": [FEATURE_LABELS[x] for x in FEATURES], "Meaning": [FEATURE_DESCRIPTIONS[x] for x in FEATURES]})
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

elif page == "Score Predictor":
    st.markdown('<div class="section-label">Live prediction engine</div>', unsafe_allow_html=True)
    st.markdown("### Build the Powerplay profile")
    st.caption("Adjust the six inputs and the forecast updates instantly.")

    c1, c2, c3 = st.columns(3)
    with c1:
        pp_runs = st.slider("🏃 Powerplay runs", 0, 90, 48, help="Total runs scored in the first six overs.")
        pp_fours = st.slider("4️⃣ Fours", 0, 18, 6)
    with c2:
        pp_sixes = st.slider("6️⃣ Sixes", 0, 10, 1)
        pp_dots = st.slider("• Dot balls", 0, 36, 16)
    with c3:
        pp_wickets = st.slider("🎯 Wickets", 0, 6, 1)
        pp_extras = st.slider("➕ Extras", 0, 15, 2)

    values = np.array([[pp_runs, pp_fours, pp_sixes, pp_dots, pp_wickets, pp_extras]], dtype=float)
    scaled_values = scaler.transform(values)
    prediction = max(0.0, float(model.predict(scaled_values)[0]))
    low = max(0.0, prediction - rmse)
    high = prediction + rmse

    left, right = st.columns([1.05, 1.35])
    with left:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " runs", "font": {"size": 38, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": "#64748b"},
                "bar": {"color": "#38bdf8", "thickness": .28},
                "bgcolor": "rgba(15,23,42,.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 130], "color": "rgba(148,163,184,.10)"},
                    {"range": [130, 180], "color": "rgba(56,189,248,.10)"},
                    {"range": [180, 230], "color": "rgba(52,211,153,.10)"},
                    {"range": [230, 300], "color": "rgba(251,146,60,.10)"},
                ],
            },
            title={"text": "Projected first-innings score", "font": {"size": 17, "color": "#cbd5e1"}},
        ))
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color":"#f8fafc"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="mini-card"><div class="mini-title">RMSE-based planning band</div><div class="mini-value">{low:.0f} – {high:.0f} runs</div></div>', unsafe_allow_html=True)
        st.caption("The range is a simple prediction-error band based on the model RMSE, not a statistical confidence interval.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Powerplay profile</div>', unsafe_allow_html=True)
        radar = go.Figure()
        raw = [pp_runs / 90 * 100, pp_fours / 18 * 100, pp_sixes / 10 * 100, (36 - pp_dots) / 36 * 100, (6 - pp_wickets) / 6 * 100, pp_extras / 15 * 100]
        radar.add_trace(go.Scatterpolar(r=raw + [raw[0]], theta=["Runs", "Fours", "Sixes", "Low dot pressure", "Wicket retention", "Extras", "Runs"], fill="toself", line={"color":"#38bdf8", "width":2}, fillcolor="rgba(56,189,248,.14)"))
        radar.update_layout(template="plotly_dark", polar={"bgcolor":"rgba(0,0,0,0)", "radialaxis":{"visible":True,"range":[0,100],"gridcolor":"rgba(148,163,184,.14)"}, "angularaxis":{"gridcolor":"rgba(148,163,184,.10)"}}, showlegend=False, height=360, margin=dict(l=35,r=35,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

        # Model contribution view in prediction space.
        contributions = scaled_values[0] * np.asarray(model.coef_)
        contrib_df = pd.DataFrame({"Feature": [FEATURE_LABELS[x] for x in FEATURES], "Contribution": contributions})
        contrib_df["Direction"] = np.where(contrib_df["Contribution"] >= 0, "Positive", "Negative")
        fig = px.bar(contrib_df.sort_values("Contribution"), x="Contribution", y="Feature", color="Direction", orientation="h", color_discrete_map={"Positive":"#34d399", "Negative":"#fb7185"})
        fig.update_layout(template="plotly_dark", height=290, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="Model signal by feature", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">Scenario presets</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    presets = {
        "Conservative": [38, 4, 0, 21, 2, 2],
        "Balanced": [48, 6, 1, 16, 1, 2],
        "Aggressive": [62, 8, 3, 10, 0, 3],
    }
    for col, (name, preset) in zip([p1, p2, p3], presets.items()):
        with col:
            vals = np.array([preset], dtype=float)
            pred = max(0.0, float(model.predict(scaler.transform(vals))[0]))
            st.markdown(f'<div class="mini-card"><div class="mini-title">{name}</div><div class="mini-value">{pred:.0f} projected runs</div><div style="color:#64748b;font-size:.72rem">{preset[0]} PP runs • {preset[2]} sixes • {preset[4]} wickets</div></div>', unsafe_allow_html=True)

elif page == "EDA Explorer":
    st.markdown('<div class="section-label">Exploration lab</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select a metric", FEATURES + ["final_score"], format_func=lambda x: FEATURE_LABELS.get(x, "Final score"))
    a, b = st.columns(2)
    with a:
        fig = px.histogram(df, x=selected, nbins=26, marginal="box", title=f"Distribution • {FEATURE_LABELS.get(selected, selected)}")
        fig.update_layout(template="plotly_dark", height=390, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = px.scatter(df, x=selected, y="final_score", color="season", hover_data=["batting_team", "bowling_team", "venue"], title=f"{FEATURE_LABELS.get(selected, selected)} vs final score")
        fig.update_layout(template="plotly_dark", height=390, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">Correlation matrix</div>', unsafe_allow_html=True)
    corr = df[FEATURES + ["final_score"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

elif page == "Model Evaluation":
    st.markdown('<div class="section-label">Model diagnostics</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("MSE", f"{float(results['mse']):.2f}")
    b.metric("RMSE", f"{rmse:.2f} runs")
    c.metric("R²", f"{r2:.3f}")

    left, right = st.columns([1.35, 1])
    with left:
        fig = px.scatter(df, x="final_score", y="predicted_final_score", opacity=.62, color="powerplay_runs", color_continuous_scale="Turbo", hover_data=["season", "batting_team", "venue"], labels={"final_score":"Actual score", "predicted_final_score":"Predicted score", "powerplay_runs":"PP runs"}, title="Actual vs predicted")
        min_v = min(df["final_score"].min(), df["predicted_final_score"].min())
        max_v = max(df["final_score"].max(), df["predicted_final_score"].max())
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line={"dash":"dash", "color":"#f8fafc", "width":1})
        fig.update_layout(template="plotly_dark", height=460, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        residuals = df["final_score"] - df["predicted_final_score"]
        fig = px.histogram(x=residuals, nbins=28, title="Residual distribution", labels={"x":"Actual − predicted"})
        fig.add_vline(x=0, line_dash="dash", line_color="#f8fafc")
        fig.update_layout(template="plotly_dark", height=460, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">Training snapshot</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Metric": ["Raw rows", "Processed matches", "Training samples", "Testing samples", "Features", "Model"],
        "Value": [results.get("raw_rows", "N/A"), results.get("processed_rows", len(df)), results.get("train_samples", "N/A"), results.get("test_samples", "N/A"), len(FEATURES), "Linear Regression"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Data Explorer":
    st.markdown('<div class="section-label">Match data explorer</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        seasons = sorted(df["season"].dropna().astype(str).unique().tolist())
        season_filter = st.multiselect("Season", seasons, default=[])
    with f2:
        teams = sorted(df["batting_team"].dropna().unique().tolist())
        team_filter = st.multiselect("Batting team", teams, default=[])
    with f3:
        min_score, max_score = st.slider("Final score", 0, 350, (50, 300))

    filtered = df[df["final_score"].between(min_score, max_score)].copy()
    if season_filter:
        filtered = filtered[filtered["season"].astype(str).isin(season_filter)]
    if team_filter:
        filtered = filtered[filtered["batting_team"].isin(team_filter)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} processed matches")
    st.dataframe(filtered, use_container_width=True, height=510)
    st.download_button("⬇ Download filtered dataset", filtered.to_csv(index=False).encode("utf-8"), file_name="IPL_ML_Processed_Filtered.csv", mime="text/csv")

st.markdown("<div style='text-align:center;color:#475569;font-size:.72rem;padding:28px 0 8px'>IPL Score Intelligence • Built with Python, Pandas, Scikit-learn, Plotly & Streamlit</div>", unsafe_allow_html=True)

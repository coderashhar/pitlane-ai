import streamlit as st
import pandas as pd
import joblib
import json
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitLane AI — F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --f1-red: #E10600;
    --f1-red-glow: rgba(225, 6, 0, 0.6);
    --f1-dark: #0D0D0D;
    --f1-carbon: #141414;
    --f1-surface: #1A1A1A;
    --f1-surface-light: #242424;
    --f1-border: #2A2A2A;
    --f1-text: #F0F0F0;
    --f1-text-muted: #888888;
    --f1-green: #00D26A;
    --f1-green-glow: rgba(0, 210, 106, 0.5);
    --f1-yellow: #FFC107;
    --f1-blue: #3D8BFF;
}

/* ── Global Reset ── */
.stApp {
    background:
        repeating-linear-gradient(
            45deg,
            rgba(255,255,255,0.015) 0px,
            rgba(255,255,255,0.015) 1px,
            transparent 1px,
            transparent 6px
        ),
        radial-gradient(ellipse at 20% 0%, rgba(225,6,0,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(225,6,0,0.05) 0%, transparent 50%),
        #0D0D0D !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }



/* ── Animated red racing stripe at top ── */
.racing-stripe {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, var(--f1-red), transparent);
    background-size: 200% 100%;
    animation: stripe-move 3s linear infinite;
    z-index: 999;
}
@keyframes stripe-move {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--f1-red), #ff3333);
    color: white;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 50px;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px var(--f1-red-glow);
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.2rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 2px;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 40px rgba(255,255,255,0.1);
}
.hero-title .accent {
    color: var(--f1-red);
    text-shadow: 0 0 30px var(--f1-red-glow);
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--f1-text-muted);
    margin-top: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.hero-divider {
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--f1-red), transparent);
    margin: 1.5rem auto;
    border-radius: 2px;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1rem 0;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--f1-red);
    text-shadow: 0 0 10px var(--f1-red-glow);
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: var(--f1-text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 2px;
}

/* ── Section Card ── */
.section-card {
    background: linear-gradient(145deg, var(--f1-surface), var(--f1-carbon));
    border: 1px solid var(--f1-border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--f1-red), transparent);
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--f1-red);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    width: 8px;
    height: 8px;
    background: var(--f1-red);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px var(--f1-red-glow);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--f1-red-glow); }
    50% { opacity: 0.5; box-shadow: 0 0 4px var(--f1-red-glow); }
}

/* ── Streamlit Widget Overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: var(--f1-surface-light) !important;
    border: 1px solid var(--f1-border) !important;
    color: var(--f1-text) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--f1-red) !important;
    box-shadow: 0 0 12px var(--f1-red-glow) !important;
}

/* Label styling */
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--f1-text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Slider styling */
.stSlider > div > div > div > div {
    background-color: var(--f1-red) !important;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--f1-red) !important;
}

/* ── Predict Button ── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--f1-red), #cc0000) !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px var(--f1-red-glow) !important;
    position: relative;
    overflow: hidden;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px var(--f1-red-glow) !important;
    background: linear-gradient(135deg, #ff1a1a, var(--f1-red)) !important;
}
.stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Boxes ── */
.result-pit {
    background: linear-gradient(145deg, rgba(225, 6, 0, 0.12), rgba(225, 6, 0, 0.05));
    border: 1px solid rgba(225, 6, 0, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: result-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-pit::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--f1-red), transparent);
    animation: stripe-move 2s linear infinite;
}
.result-nopit {
    background: linear-gradient(145deg, rgba(0, 210, 106, 0.12), rgba(0, 210, 106, 0.05));
    border: 1px solid rgba(0, 210, 106, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: result-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-nopit::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--f1-green), transparent);
}
@keyframes result-appear {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 4px;
    margin-bottom: 0.5rem;
}
.result-pit .result-title { color: var(--f1-red); text-shadow: 0 0 20px var(--f1-red-glow); }
.result-nopit .result-title { color: var(--f1-green); text-shadow: 0 0 20px var(--f1-green-glow); }
.result-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--f1-text-muted);
    font-weight: 400;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--f1-carbon) !important;
    border-right: 1px solid var(--f1-border) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--f1-surface-light), var(--f1-surface)) !important;
    border: 1px solid var(--f1-border) !important;
    color: var(--f1-text) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.25rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    margin-bottom: 0.5rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--f1-red) !important;
    box-shadow: 0 0 15px var(--f1-red-glow) !important;
    transform: translateX(4px) !important;
}

/* ── Telemetry display ── */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 1rem;
}
.telemetry-cell {
    background: var(--f1-surface-light);
    border: 1px solid var(--f1-border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: border-color 0.3s ease;
}
.telemetry-cell:hover {
    border-color: var(--f1-red);
}
.telemetry-cell .tel-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: var(--f1-text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.telemetry-cell .tel-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--f1-text);
    margin-top: 4px;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--f1-border);
    margin-top: 2rem;
}
.footer-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--f1-text-muted);
    letter-spacing: 0.5px;
}
.footer-brand {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    color: var(--f1-red);
}

/* ── Tyre compound badges ── */
.tyre-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.tyre-soft { background: rgba(255, 0, 0, 0.15); color: #FF4444; border: 1px solid rgba(255, 0, 0, 0.3); }
.tyre-medium { background: rgba(255, 193, 7, 0.15); color: #FFC107; border: 1px solid rgba(255, 193, 7, 0.3); }
.tyre-hard { background: rgba(255, 255, 255, 0.1); color: #CCCCCC; border: 1px solid rgba(255, 255, 255, 0.2); }
.tyre-inter { background: rgba(0, 180, 0, 0.15); color: #44CC44; border: 1px solid rgba(0, 180, 0, 0.3); }
.tyre-wet { background: rgba(0, 100, 255, 0.15); color: #4488FF; border: 1px solid rgba(0, 100, 255, 0.3); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .stats-bar { gap: 1.5rem; }
    .stat-value { font-size: 1.2rem; }
}

/* ── Expander styling ── */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--f1-text-muted) !important;
    background-color: var(--f1-surface) !important;
    border-radius: 10px !important;
}

/* ── Subheader override ── */
.stForm h3, h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--f1-red) !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: var(--f1-surface);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--f1-border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--f1-text-muted) !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--f1-red), #cc0000) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--f1-red-glow) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: var(--f1-red) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--f1-surface-light) !important;
    border: 1px solid var(--f1-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--f1-text) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Racing stripe ────────────────────────────────────────────────────────────
st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

# ─── Load Model & Encoders ────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('best_f1_pit_stop_model.pkl')
    with open('encoders.json', 'r') as f:
        encoders = json.load(f)
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model assets: {e}")
    st.stop()

# ─── Session State Defaults ───────────────────────────────────────────────────
defaults = {
    'driver': encoders['Driver'][0],
    'lap_number': 10,
    'compound': encoders['Compound'][0],
    'stint': 1,
    'tyre_life': 5.0,
    'position': 1,
    'lap_time': 90.0,
    'race': encoders['Race'][0],
    'lap_time_delta': 0.0,
    'cumulative_deg': 1.5,
    'race_progress': 0.2,
    'norm_tyre_life': 0.3,
    'position_change': 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Sidebar — Simulation Scenarios ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="font-family: 'Orbitron', monospace; font-size: 0.7rem; font-weight: 700; 
                    color: #E10600; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.5rem;">
            ⚡ Quick Scenarios
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #888; margin-bottom: 1rem;">
            Auto-fill telemetry with real-world race data for instant simulation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔴  Worn Tyres — Pit Likely", use_container_width=True):
        st.session_state.update({
            'driver': 'ALB', 'lap_number': 16, 'compound': 'MEDIUM',
            'stint': 1, 'tyre_life': 17.0, 'position': 12,
            'lap_time': 94.282, 'race': 'Abu Dhabi Grand Prix',
            'lap_time_delta': 2.553, 'cumulative_deg': -6.343,
            'race_progress': 0.27, 'norm_tyre_life': 1.0, 'position_change': -1,
        })
        st.rerun()
        
    if st.button("🟢  Fresh Tyres — Stay Out", use_container_width=True):
        st.session_state.update(defaults)
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #555; line-height: 1.6;">
        <strong style="color: #888;">How it works</strong><br>
        This model uses XGBoost trained on real F1 telemetry data to predict 
        whether a driver will make a pit stop on the next lap based on tyre 
        degradation, race position, and pace metrics.
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Machine Learning Powered</div>
    <h1 class="hero-title">PITLANE <span class="accent">AI</span></h1>
    <p class="hero-subtitle">Real-time F1 pit stop prediction using race telemetry & machine learning</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Bar ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{len(encoders['Driver'])}</div>
        <div class="stat-label">Drivers</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{len(encoders['Race'])}</div>
        <div class="stat-label">Circuits</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{len(encoders['Compound'])}</div>
        <div class="stat-label">Compounds</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">15</div>
        <div class="stat-label">Features</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Form ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    
    # ── Tab-based Input Sections ──
    tab1, tab2, tab3 = st.tabs(["🏁  RACE & DRIVER", "🛞  TYRE DATA", "⏱️  PACE & PROGRESS"])
    
    with tab1:
        st.markdown('<div class="section-label">Driver & Race Configuration</div>', unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2, gap="large")
        with col_d1:
            driver = st.selectbox(
                "Driver Code",
                encoders['Driver'],
                index=encoders['Driver'].index(st.session_state['driver']),
                help="Three-letter driver abbreviation (e.g., VER, HAM, LEC)"
            )
            position = st.number_input(
                "Grid Position",
                min_value=1, max_value=20,
                value=st.session_state['position'],
                help="Current race position (1-20)"
            )
        with col_d2:
            race = st.selectbox(
                "Grand Prix",
                encoders['Race'],
                index=encoders['Race'].index(st.session_state['race']),
                help="Select the Grand Prix circuit"
            )
            position_change = st.number_input(
                "Position Change",
                min_value=-20, max_value=20,
                value=st.session_state['position_change'],
                help="Positions gained (+) or lost (-) this stint"
            )
    
    with tab2:
        st.markdown('<div class="section-label">Tyre Telemetry & Degradation</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2, gap="large")
        with col_t1:
            compound = st.selectbox(
                "Tyre Compound",
                encoders['Compound'],
                index=encoders['Compound'].index(st.session_state['compound']),
                help="Current tyre compound fitted"
            )
            tyre_life = st.number_input(
                "Tyre Life (Laps)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state['tyre_life']),
                help="Number of laps on the current set of tyres"
            )
            cumulative_deg = st.number_input(
                "Cumulative Degradation",
                value=float(st.session_state['cumulative_deg']),
                help="Total accumulated tyre degradation factor"
            )
        with col_t2:
            stint = st.number_input(
                "Current Stint",
                min_value=1, max_value=10,
                value=st.session_state['stint'],
                help="Which stint the driver is currently on"
            )
            normalized_tyre_life = st.slider(
                "Normalized Tyre Life",
                0.0, 1.0,
                value=float(st.session_state['norm_tyre_life']),
                help="Tyre life normalized between 0 (fresh) and 1 (end of life)"
            )
    
    with tab3:
        st.markdown('<div class="section-label">Lap Pace & Race Progress</div>', unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns(3, gap="large")
        with col_p1:
            lap_number = st.number_input(
                "Lap Number",
                min_value=1, max_value=100,
                value=st.session_state['lap_number'],
                help="Current lap number"
            )
        with col_p2:
            lap_time = st.number_input(
                "Lap Time (seconds)",
                min_value=50.0, max_value=200.0,
                value=float(st.session_state['lap_time']),
                help="Last recorded lap time in seconds"
            )
        with col_p3:
            lap_time_delta = st.number_input(
                "Lap Delta (seconds)",
                min_value=-100.0, max_value=100.0,
                value=float(st.session_state['lap_time_delta']),
                help="Time delta compared to previous lap"
            )
        
        race_progress = st.slider(
            "Race Progress",
            0.0, 1.0,
            value=float(st.session_state['race_progress']),
            help="Fraction of the race completed (0.0 = start, 1.0 = finish)"
        )
    
    # ── Submit ──
    st.markdown("")
    submitted = st.form_submit_button("🏁  ANALYZE TELEMETRY", use_container_width=True)

# ─── Telemetry Summary (always visible) ───────────────────────────────────────
with st.expander("📊 Current Telemetry Overview", expanded=False):
    # Tyre compound badge class
    compound_class = {
        'SOFT': 'tyre-soft', 'MEDIUM': 'tyre-medium', 'HARD': 'tyre-hard',
        'INTERMEDIATE': 'tyre-inter', 'WET': 'tyre-wet'
    }.get(st.session_state['compound'], 'tyre-hard')
    
    st.markdown(f"""
    <div class="telemetry-grid">
        <div class="telemetry-cell">
            <div class="tel-label">Driver</div>
            <div class="tel-value">{st.session_state['driver']}</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Position</div>
            <div class="tel-value">P{st.session_state['position']}</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Lap</div>
            <div class="tel-value">{st.session_state['lap_number']}</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Compound</div>
            <div class="tel-value"><span class="tyre-badge {compound_class}">{st.session_state['compound']}</span></div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Tyre Life</div>
            <div class="tel-value">{st.session_state['tyre_life']}L</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Stint</div>
            <div class="tel-value">S{st.session_state['stint']}</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Lap Time</div>
            <div class="tel-value">{st.session_state['lap_time']}s</div>
        </div>
        <div class="telemetry-cell">
            <div class="tel-label">Progress</div>
            <div class="tel-value">{int(st.session_state['race_progress'] * 100)}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame({
        'Driver': [encoders['Driver'].index(driver)],
        'LapNumber': [lap_number],
        'Compound': [encoders['Compound'].index(compound)],
        'Stint': [stint],
        'TyreLife': [tyre_life],
        'Position': [position],
        'LapTime (s)': [lap_time],
        'Race': [encoders['Race'].index(race)],
        'LapTime_Delta': [lap_time_delta],
        'Cumulative_Degradation': [cumulative_deg],
        'RaceProgress': [race_progress],
        'Normalized_TyreLife': [normalized_tyre_life],
        'Position_Change': [position_change],
        'LapTime_Rolling_3': [lap_time],
        'LapTime_Gradient': [lap_time_delta],
    })
    
    with st.spinner("Analyzing telemetry data..."):
        time.sleep(0.8)  # Brief delay for dramatic effect
        prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.markdown("""
        <div class="result-pit">
            <div class="result-icon">🔴</div>
            <div class="result-title">BOX BOX BOX</div>
            <div class="result-subtitle">High probability of pit stop on the next lap — prepare the crew</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-nopit">
            <div class="result-icon">🟢</div>
            <div class="result-title">STAY OUT</div>
            <div class="result-subtitle">Driver is likely to continue — tyres and pace are within acceptable range</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-text">
        Built with XGBoost & Real F1 Telemetry Data &nbsp;·&nbsp; 
        <span class="footer-brand">PITLANE AI</span> &nbsp;·&nbsp; 
        Powered by Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

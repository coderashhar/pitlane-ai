import streamlit as st
import pandas as pd
import joblib
import json

# Set page config for aesthetics
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E10600; /* F1 Red */
        text-align: center;
        margin-bottom: 2rem;
    }
    .predict-btn {
        background-color: #E10600;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        width: 100%;
        text-align: center;
        transition: 0.3s;
    }
    .predict-btn:hover {
        background-color: #b30500;
    }
    .result-box-pit {
        background-color: rgba(225, 6, 0, 0.2);
        border: 2px solid #E10600;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff4d4d;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(225, 6, 0, 0.5);
    }
    .result-box-nopit {
        background-color: rgba(0, 200, 0, 0.2);
        border: 2px solid #00c800;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #4dff4d;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(0, 200, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Load Model and Encoders
@st.cache_resource
def load_assets():
    model = joblib.load('best_f1_pit_stop_model.pkl')
    with open('encoders.json', 'r') as f:
        encoders = json.load(f)
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model or encoders. Make sure you have trained the model. {e}")
    st.stop()

# Initialize session state for default values
if 'driver' not in st.session_state:
    st.session_state['driver'] = encoders['Driver'][0]
    st.session_state['lap_number'] = 10
    st.session_state['compound'] = encoders['Compound'][0]
    st.session_state['stint'] = 1
    st.session_state['tyre_life'] = 5.0
    st.session_state['position'] = 1
    st.session_state['lap_time'] = 90.0
    st.session_state['race'] = encoders['Race'][0]
    st.session_state['lap_time_delta'] = 0.0
    st.session_state['cumulative_deg'] = 1.5
    st.session_state['race_progress'] = 0.2
    st.session_state['norm_tyre_life'] = 0.3
    st.session_state['position_change'] = 0

# Sidebar for scenarios
with st.sidebar:
    st.header("Simulation Scenarios")
    st.write("Click below to auto-fill the form with values typical for a pit stop.")
    if st.button("🚨 Simulate Worn Tyres (Pit Stop)"):
        # Uses exact telemetry from Albon at Abu Dhabi just before a pit stop
        st.session_state['driver'] = 'ALB'
        st.session_state['lap_number'] = 16
        st.session_state['compound'] = 'MEDIUM'
        st.session_state['stint'] = 1
        st.session_state['tyre_life'] = 17.0
        st.session_state['position'] = 12
        st.session_state['lap_time'] = 94.282
        st.session_state['race'] = 'Abu Dhabi Grand Prix'
        st.session_state['lap_time_delta'] = 2.553
        st.session_state['cumulative_deg'] = -6.343
        st.session_state['race_progress'] = 0.27
        st.session_state['norm_tyre_life'] = 1.0
        st.session_state['position_change'] = -1
        st.rerun()
    if st.button("🟢 Simulate Fresh Tyres (Stay Out)"):
        st.session_state['driver'] = encoders['Driver'][0]
        st.session_state['lap_number'] = 10
        st.session_state['compound'] = encoders['Compound'][0]
        st.session_state['stint'] = 1
        st.session_state['tyre_life'] = 5.0
        st.session_state['position'] = 1
        st.session_state['lap_time'] = 90.0
        st.session_state['race'] = encoders['Race'][0]
        st.session_state['lap_time_delta'] = 0.0
        st.session_state['cumulative_deg'] = 1.5
        st.session_state['race_progress'] = 0.2
        st.session_state['norm_tyre_life'] = 0.3
        st.session_state['position_change'] = 0
        st.rerun()

st.markdown('<div class="main-header">🏁 F1 Pit Stop Predictor</div>', unsafe_allow_html=True)
st.write("Enter the telemetry and race conditions to predict if the driver will pit in the next lap.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Driver Info")
        driver = st.selectbox("Driver", encoders['Driver'], index=encoders['Driver'].index(st.session_state['driver']))
        race = st.selectbox("Race Location", encoders['Race'], index=encoders['Race'].index(st.session_state['race']))
        position = st.number_input("Current Position", min_value=1, max_value=20, value=st.session_state['position'])
        position_change = st.number_input("Position Change", min_value=-20, max_value=20, value=st.session_state['position_change'])
        
    with col2:
        st.subheader("Tyre Info")
        compound = st.selectbox("Tyre Compound", encoders['Compound'], index=encoders['Compound'].index(st.session_state['compound']))
        stint = st.number_input("Current Stint", min_value=1, max_value=10, value=st.session_state['stint'])
        tyre_life = st.number_input("Tyre Life (Laps)", min_value=0.0, max_value=100.0, value=float(st.session_state['tyre_life']))
        normalized_tyre_life = st.slider("Normalized Tyre Life", 0.0, 1.0, value=float(st.session_state['norm_tyre_life']))
        cumulative_deg = st.number_input("Cumulative Degradation", value=float(st.session_state['cumulative_deg']))

    st.subheader("Pace & Progress")
    col3, col4, col5 = st.columns(3)
    with col3:
        lap_number = st.number_input("Lap Number", min_value=1, max_value=100, value=st.session_state['lap_number'])
    with col4:
        lap_time = st.number_input("Lap Time (s)", min_value=50.0, max_value=200.0, value=float(st.session_state['lap_time']))
    with col5:
        lap_time_delta = st.number_input("Lap Time Delta (s)", min_value=-100.0, max_value=100.0, value=float(st.session_state['lap_time_delta']))
        
    race_progress = st.slider("Race Progress", 0.0, 1.0, value=float(st.session_state['race_progress']))

    submit_button = st.form_submit_button(label='Predict Next Lap Pit Stop')

if submit_button:
    # Prepare input dataframe
    # Features: ['Driver', 'LapNumber', 'Compound', 'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'Race', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Normalized_TyreLife', 'Position_Change', 'LapTime_Rolling_3', 'LapTime_Gradient']
    
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
        'LapTime_Rolling_3': [lap_time],        # Approximated for simulation
        'LapTime_Gradient': [lap_time_delta]    # Approximated for simulation
    })
    
    # Predict
    with st.spinner("Analyzing telemetry..."):
        prediction = model.predict(input_data)[0]
        
    if prediction == 1:
        st.markdown('<div class="result-box-pit">🏎️ BOX BOX BOX! <br>The driver is highly likely to pit on the next lap.</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="result-box-nopit">🟢 STAY OUT. <br>The driver is likely to continue for another lap.</div>', unsafe_allow_html=True)

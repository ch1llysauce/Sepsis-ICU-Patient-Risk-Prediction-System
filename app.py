import os
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

from src.sepsis_model import (
    train_sepsis_bayesian_network,
    query_sepsis_risk,
    discretize_single_vitals,
    render_sepsis_dag
)

# Page Setup
st.set_page_config(
    page_title="ICU Sepsis Early Warning System",
    page_icon="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/hospital.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------------------------------
# CLINICAL DARK-MODE CSS WITH BOOTSTRAP ICONS
# -------------------------------------------------------------------------------------------------
st.html("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header */
.med-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
}
.med-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 14px;
    border-radius: 9999px;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.35);
    color: #38bdf8;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.med-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}
.med-sub {
    font-size: 0.94rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

/* Risk Alert Hero Cards */
.risk-card {
    border-radius: 16px;
    padding: 22px 28px;
    border: 1.5px solid;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
}
.risk-stable {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(52, 211, 153, 0.45);
}
.risk-warning {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(251, 146, 60, 0.45);
}
.risk-emergency {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(248, 113, 113, 0.55);
}

.risk-layout {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}
.risk-left {
    display: flex;
    align-items: center;
    gap: 18px;
}
.risk-icon {
    font-size: 2.5rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
.risk-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}
.risk-headline {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 3px;
}
.risk-stable .risk-headline { color: #34d399; }
.risk-warning .risk-headline { color: #fb923c; }
.risk-emergency .risk-headline { color: #f87171; }

.risk-stat {
    display: flex;
    flex-direction: column;
}
.risk-stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.risk-stat-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 2px;
}
.risk-stat-sub {
    font-size: 0.8rem;
    color: #38bdf8;
    font-weight: 600;
}

/* Action Box */
.action-box {
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.9rem;
    color: #f1f5f9;
}
.action-badge {
    font-weight: 800;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.12);
    color: #f8fafc;
    white-space: nowrap;
}

/* Vital Signs Card Grid */
.vital-card {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.18s ease;
}
.vital-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
}
.vital-title {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.vital-num {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 0 2px 0;
}
.vital-normal-range {
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 8px;
}
.vital-pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    align-self: flex-start;
}
.v-normal { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.v-warning { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
.v-danger  { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Sidebar */
div[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Prevent UI dimming / fading during script reruns */
[data-testid="stAppViewBlockContainer"] {
    transition: none !important;
}
.stApp[data-test-script-state="running"] [data-testid="stAppViewBlockContainer"] {
    opacity: 1 !important;
}

/* -----------------------------------------------------------
   MOBILE & TABLET RESPONSIVE ADAPTATIONS
   ----------------------------------------------------------- */
@media (max-width: 768px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 12px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .med-header {
        padding: 16px 18px;
    }
    .med-title {
        font-size: 1.4rem;
    }
    .med-sub {
        font-size: 0.85rem;
    }
    .risk-card {
        padding: 16px 18px;
    }
    .risk-layout {
        flex-direction: column;
        align-items: flex-start;
        gap: 14px;
    }
    .risk-headline {
        font-size: 1.35rem;
    }
    .risk-stat {
        width: 100%;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 8px;
    }
    .action-box {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }
    .action-badge {
        align-self: flex-start;
    }
    .vital-card {
        margin-bottom: 8px;
    }
}

@media (min-width: 769px) and (max-width: 1100px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 12px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 calc(50% - 12px) !important;
        min-width: calc(50% - 12px) !important;
        margin-bottom: 10px;
    }
}
</style>
""")

# -------------------------------------------------------------------------------------------------
# LOAD TRAINED SEPSIS BAYESIAN NETWORK
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_sepsis_network():
    csv_path = os.path.join("data", "physionet_sepsis_patients.csv")
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please run `python src/download_physionet.py`.")
        st.stop()
    df_raw = pd.read_csv(csv_path)
    model, infer, df_bn = train_sepsis_bayesian_network(df_raw)
    return model, infer, df_raw

bn_model, bn_infer, df_patients = load_sepsis_network()

# -------------------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DATASETS
# -------------------------------------------------------------------------------------------------
st.sidebar.html("""
<div style="padding: 10px 0 12px 0;">
    <div style="font-size: 0.72rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 6px;">
        <i class="bi bi-cpu"></i> Clinical Bayesian AI
    </div>
    <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; line-height: 1.2; margin-top: 4px;">
        ICU Sepsis Prediction
    </div>
</div>
""")

mode = st.sidebar.radio(
    "Monitoring Mode:",
    [
        "Live Bedside Vitals Simulator",
        "PhysioNet ICU Patient Records",
        "Clinical Causal Map (DAG)"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(f"""
**Dataset Provenance:**
- **Source:** PhysioNet Challenge 2019
- **Cohort Size:** 290 ICU Patients
- **Hourly Records:** 11,051 Clinical Hours
- **Baseline Sepsis Rate:** ~2.3%
""")

# -------------------------------------------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------------------------------------------
st.html("""
<div class="med-header">
    <div class="med-tag"><i class="bi bi-hospital"></i> Intensive Care Unit (ICU) Decision Support</div>
    <div class="med-title">Early Sepsis Detection Bayesian Network</div>
    <div class="med-sub">
        Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. 
        Because symptoms develop gradually, this <b>Causal Bayesian Network</b> computes real-time sepsis risk 
        from physiological vitals (Heart Rate, Blood Pressure, Temperature, Respiration, Oxygen) before septic shock occurs.
    </div>
</div>
""")

# -------------------------------------------------------------------------------------------------
# MODE 1: LIVE BEDSIDE VITALS SIMULATOR
# -------------------------------------------------------------------------------------------------
if mode == "Live Bedside Vitals Simulator":
    # Initialize session state for bedside vitals
    if 'hr' not in st.session_state:
        st.session_state.hr = 72
        st.session_state.temp = 36.8
        st.session_state.resp = 15
        st.session_state.map = 82
        st.session_state.o2 = 98
        st.session_state.map_on = True
        st.session_state.o2_on = True
        st.session_state.bed_name = "Bed 1: Normal Recovery Patient"

    st.markdown("##### <i class='bi bi-lightning-charge'></i> Quick Patient Scenarios (Bedside Presets):", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)

    if p1.button("Bed 1: Normal Recovery Patient", use_container_width=True):
        st.session_state.hr = 72; st.session_state.temp = 36.8; st.session_state.resp = 15
        st.session_state.map = 82; st.session_state.o2 = 98
        st.session_state.map_on = True; st.session_state.o2_on = True
        st.session_state.bed_name = "Bed 1: Normal Recovery Patient"
        st.rerun()

    if p2.button("Bed 2: Fever & High Pulse", use_container_width=True):
        st.session_state.hr = 108; st.session_state.temp = 38.6; st.session_state.resp = 22
        st.session_state.map = 74; st.session_state.o2 = 96
        st.session_state.map_on = True; st.session_state.o2_on = True
        st.session_state.bed_name = "Bed 2: Fever & High Pulse (Developing Infection)"
        st.rerun()

    if p3.button("Bed 3: Critical Septic Shock", use_container_width=True):
        st.session_state.hr = 125; st.session_state.temp = 39.2; st.session_state.resp = 28
        st.session_state.map = 52; st.session_state.o2 = 91
        st.session_state.map_on = True; st.session_state.o2_on = True
        st.session_state.bed_name = "Bed 3: Critical Septic Shock (Hypotensive Crisis)"
        st.rerun()

    if p4.button("Bed 4: Missing Sensor Telemetry", use_container_width=True):
        st.session_state.hr = 115; st.session_state.temp = 38.9; st.session_state.resp = 26
        st.session_state.map = 80; st.session_state.o2 = 98
        st.session_state.map_on = False; st.session_state.o2_on = False
        st.session_state.bed_name = "Bed 4: Missing Sensor Failure (Partial Telemetry)"
        st.rerun()

    st.caption(f"Simulating Bedside Telemetry: **{st.session_state.bed_name}**")

    st.markdown("##### <i class='bi bi-sliders'></i> Bedside Patient Monitor Controls:", unsafe_allow_html=True)
    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)

    with col_v1:
        hr_val = st.number_input("Heart Rate (bpm):", min_value=30, max_value=200, value=int(st.session_state.hr))
        st.session_state.hr = hr_val
    with col_v2:
        temp_val = st.number_input("Temperature (°C):", min_value=32.0, max_value=42.0, value=float(st.session_state.temp), step=0.1)
        st.session_state.temp = temp_val
    with col_v3:
        resp_val = st.number_input("Respiration Rate (br/min):", min_value=8, max_value=50, value=int(st.session_state.resp))
        st.session_state.resp = resp_val
    with col_v4:
        map_val = st.number_input("Blood Pressure MAP (mmHg):", min_value=30, max_value=150, value=int(st.session_state.map))
        st.session_state.map = map_val
        map_active = st.checkbox("BP Sensor Active", value=st.session_state.map_on)
        st.session_state.map_on = map_active
    with col_v5:
        o2_val = st.number_input("Oxygen SpO2 (%):", min_value=70, max_value=100, value=int(st.session_state.o2))
        st.session_state.o2 = o2_val
        o2_active = st.checkbox("O2 Sensor Active", value=st.session_state.o2_on)
        st.session_state.o2_on = o2_active

    # Convert to Bayesian Evidence
    patient_ev = discretize_single_vitals(
        hr=hr_val,
        temp=temp_val,
        resp=resp_val,
        map_bp=map_val if st.session_state.map_on else None,
        o2sat=o2_val if st.session_state.o2_on else None
    )

    # Compute Bayesian Posterior
    diagnosis = query_sepsis_risk(bn_infer, patient_ev)
    sepsis_prob = diagnosis['sepsis_probability']
    prior_prob = diagnosis['prior_probability']
    cal_score = diagnosis['calibrated_score']
    rel_risk = diagnosis['relative_risk']
    risk_level = diagnosis['alert_level']
    category = diagnosis['risk_category']
    action = diagnosis['recommendation']

    card_class = "risk-stable" if risk_level == "STABLE" else ("risk-warning" if risk_level == "WARNING" else "risk-emergency")
    
    # Clean vector icon replacement
    if risk_level == "STABLE":
        card_icon_html = '<i class="bi bi-shield-check" style="color: #34d399;"></i>'
    elif risk_level == "WARNING":
        card_icon_html = '<i class="bi bi-exclamation-triangle-fill" style="color: #fb923c;"></i>'
    else:
        card_icon_html = '<i class="bi bi-bell-fill" style="color: #f87171;"></i>'

    # Master Clinical Banner
    st.html(f"""
    <div class="risk-card {card_class}">
        <div class="risk-layout">
            <div class="risk-left">
                <div class="risk-icon">{card_icon_html}</div>
                <div>
                    <div class="risk-label">CLINICAL RISK CLASSIFICATION</div>
                    <div class="risk-headline">{category}</div>
                </div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">Clinical Sepsis Index</div>
                <div class="risk-stat-value">{cal_score:.0f}% Risk</div>
                <div class="risk-stat-sub">{rel_risk:.1f}x baseline ICU risk</div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">Bayesian Posterior Prob</div>
                <div class="risk-stat-value">{sepsis_prob * 100:.1f}%</div>
                <div class="risk-stat-sub">Baseline: {prior_prob * 100:.1f}%</div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">Monitored Vital Signs</div>
                <div class="risk-stat-value">{len(diagnosis['active_evidence'])} / 5 Active</div>
                <div class="risk-stat-sub">Continuous telemetry</div>
            </div>
        </div>
        <div class="action-box">
            <span class="action-badge">Doctor's Protocol</span>
            <span>{action}</span>
        </div>
    </div>
    """)

    # Bedside Vitals Evidence Cards
    st.markdown("##### <i class='bi bi-clipboard2-pulse'></i> Extracted Clinical Telemetry:", unsafe_allow_html=True)
    c_card1, c_card2, c_card3, c_card4, c_card5 = st.columns(5)

    with c_card1:
        hr_state = patient_ev.get('HeartRate', 'Unknown')
        pill_c = "v-normal" if hr_state == "Normal" else ("v-warning" if hr_state == "Bradycardia" else "v-danger")
        st.html(f"""
        <div class="vital-card">
            <div>
                <div class="vital-title"><i class="bi bi-heart-pulse"></i> Heart Rate (Pulse)</div>
                <div class="vital-num">{hr_val} <span style="font-size: 0.85rem; color: #94a3b8;">bpm</span></div>
                <div class="vital-normal-range">Normal: 60–100 bpm</div>
            </div>
            <span class="vital-pill {pill_c}">{hr_state}</span>
        </div>
        """)

    with c_card2:
        t_state = patient_ev.get('Temperature', 'Unknown')
        pill_c = "v-normal" if t_state == "Normal" else "v-danger"
        st.html(f"""
        <div class="vital-card">
            <div>
                <div class="vital-title"><i class="bi bi-thermometer-half"></i> Body Temp</div>
                <div class="vital-num">{temp_val:.1f} <span style="font-size: 0.85rem; color: #94a3b8;">°C</span></div>
                <div class="vital-normal-range">Normal: 36.5–37.5 °C</div>
            </div>
            <span class="vital-pill {pill_c}">{t_state}</span>
        </div>
        """)

    with c_card3:
        r_state = patient_ev.get('RespirationRate', 'Unknown')
        pill_c = "v-normal" if r_state == "Normal" else "v-danger"
        st.html(f"""
        <div class="vital-card">
            <div>
                <div class="vital-title"><i class="bi bi-lungs"></i> Breathing Rate</div>
                <div class="vital-num">{resp_val} <span style="font-size: 0.85rem; color: #94a3b8;">br/min</span></div>
                <div class="vital-normal-range">Normal: 12–20 br/min</div>
            </div>
            <span class="vital-pill {pill_c}">{r_state}</span>
        </div>
        """)

    with c_card4:
        map_state = patient_ev.get('BloodPressure_MAP', 'Sensor Disconnected')
        pill_c = "v-normal" if map_state == "Normal" else "v-danger"
        st.html(f"""
        <div class="vital-card">
            <div>
                <div class="vital-title"><i class="bi bi-droplet-half"></i> Blood Pressure (MAP)</div>
                <div class="vital-num">{map_val if st.session_state.map_on else "OFF"} <span style="font-size: 0.85rem; color: #94a3b8;">mmHg</span></div>
                <div class="vital-normal-range">Shock threshold: <65</div>
            </div>
            <span class="vital-pill {pill_c}">{map_state}</span>
        </div>
        """)

    with c_card5:
        o2_state = patient_ev.get('OxygenSaturation', 'Sensor Disconnected')
        pill_c = "v-normal" if o2_state == "Normal" else "v-danger"
        st.html(f"""
        <div class="vital-card">
            <div>
                <div class="vital-title"><i class="bi bi-activity"></i> Oxygen SpO2</div>
                <div class="vital-num">{o2_val if st.session_state.o2_on else "OFF"} <span style="font-size: 0.85rem; color: #94a3b8;">%</span></div>
                <div class="vital-normal-range">Normal: >= 95%</div>
            </div>
            <span class="vital-pill {pill_c}">{o2_state}</span>
        </div>
        """)

    st.html("<div style='height: 16px;'></div>")

    # Bayesian Belief Chart
    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        st.markdown("##### <i class='bi bi-bar-chart-line'></i> Bayesian Belief Updating: Baseline vs. After Vitals Observed", unsafe_allow_html=True)
        df_bars = pd.DataFrame([
            {'Metric': 'General ICU Baseline (Prior)', 'Probability (%)': round(prior_prob * 100, 2)},
            {'Metric': 'Current Patient Risk (Posterior)', 'Probability (%)': round(sepsis_prob * 100, 2)}
        ])
        chart = alt.Chart(df_bars).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color='#38bdf8').encode(
            y=alt.Y('Metric:N', title=None, axis=alt.Axis(labelFontSize=13, labelFontWeight='bold')),
            x=alt.X('Probability (%):Q', title='Calculated Sepsis Probability (%)', scale=alt.Scale(domain=[0, max(20, sepsis_prob*120)])),
            tooltip=['Metric:N', 'Probability (%):Q']
        ).properties(height=180)
        st.altair_chart(chart, use_container_width=True)

    with col_g2:
        st.html(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px 20px; height: 100%;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <i class="bi bi-info-circle-fill"></i> Presentation Talking Point (Missing Data):
            </div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 6px;">
                Subukan ang <b>Bed 4 (Missing Sensor Telemetry)</b> kung saan naka-disconnect ang Blood Pressure at Oxygen sensors.
            </p>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                Hindi nag-crash ang Bayesian Network! Awtomatiko nitong <b>mini-marginalize</b> ang mga nawawalang readings at binibigyan pa rin ang doktor ng tamang risk estimate.
            </p>
        </div>
        """)

# -------------------------------------------------------------------------------------------------
# MODE 2: REAL PHYSIONET ICU PATIENTS
# -------------------------------------------------------------------------------------------------
elif mode == "PhysioNet ICU Patient Records":
    st.markdown("##### Select Real ICU Patient Record (PhysioNet 2019 Benchmark):")
    
    unique_patients = df_patients['PatientID'].unique().tolist()
    
    col_p1, col_p2 = st.columns([1.5, 1])
    with col_p1:
        sel_patient = st.selectbox("Select Patient ID:", unique_patients[:100])
        
    pat_df = df_patients[df_patients['PatientID'] == sel_patient].reset_index(drop=True)
    
    with col_p2:
        hr_idx = st.slider("Hour of ICU Stay:", 1, len(pat_df), 1)

    patient_hour = pat_df.iloc[hr_idx - 1]
    
    hr_val = patient_hour['HR'] if not pd.isna(patient_hour['HR']) else 80.0
    temp_val = patient_hour['Temp'] if not pd.isna(patient_hour['Temp']) else 37.0
    resp_val = patient_hour['Resp'] if not pd.isna(patient_hour['Resp']) else 16.0
    map_val = patient_hour['MAP'] if not pd.isna(patient_hour['MAP']) else 80.0
    o2_val = patient_hour['O2Sat'] if not pd.isna(patient_hour['O2Sat']) else 98.0
    sepsis_actual = int(patient_hour['SepsisLabel'])

    pat_ev = discretize_single_vitals(hr=hr_val, temp=temp_val, resp=resp_val, map_bp=map_val, o2sat=o2_val)
    pat_diag = query_sepsis_risk(bn_infer, pat_ev)

    is_sepsis_hour = (sepsis_actual == 1)
    status_box = "risk-emergency" if is_sepsis_hour else "risk-stable"
    card_icon_rec = '<i class="bi bi-exclamation-octagon-fill" style="color: #f87171;"></i>' if is_sepsis_hour else '<i class="bi bi-check-circle-fill" style="color: #34d399;"></i>'
    
    st.html(f"""
    <div class="risk-card {status_box}">
        <div class="risk-layout">
            <div class="risk-left">
                <div class="risk-icon">{card_icon_rec}</div>
                <div>
                    <div class="risk-label">PHYSIONET CLINICAL GROUND TRUTH (HOUR {hr_idx})</div>
                    <div class="risk-headline">{'SEPSIS ONSET CONFIRMED' if is_sepsis_hour else 'NO SEPSIS ONSET'}</div>
                </div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">Bayesian Predicted Risk</div>
                <div class="risk-stat-value">{pat_diag['sepsis_probability'] * 100:.1f}%</div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">Patient Age / Gender</div>
                <div class="risk-stat-value">{int(patient_hour['Age']) if not pd.isna(patient_hour['Age']) else 60} yo / {'M' if patient_hour['Gender']==1 else 'F'}</div>
            </div>
            <div class="risk-stat">
                <div class="risk-stat-label">ICU Stay Duration</div>
                <div class="risk-stat-value">{int(patient_hour['ICULOS'])} hrs</div>
            </div>
        </div>
    </div>
    """)

    st.markdown("##### Hourly Vitals Trajectory in ICU:")
    vitals_to_plot = ['HR', 'MAP', 'Resp']
    available_cols = [c for c in vitals_to_plot if c in pat_df.columns]
    
    if available_cols:
        chart_df = pat_df[['ICULOS'] + available_cols].dropna().set_index('ICULOS')
        st.line_chart(chart_df)

# -------------------------------------------------------------------------------------------------
# MODE 3: CAUSAL DAG DIAGRAM
# -------------------------------------------------------------------------------------------------
elif mode == "Clinical Causal Map (DAG)":
    st.markdown("##### Causal Directed Acyclic Graph (DAG) for Sepsis Pathophysiology:")
    
    col_d1, col_d2 = st.columns([1.4, 1])
    with col_d1:
        dag_path = os.path.join("reports", "sepsis_dag.png")
        if os.path.exists(dag_path):
            st.image(dag_path)
        else:
            fig_dag = render_sepsis_dag()
            st.pyplot(fig_dag)
    with col_d2:
        st.markdown("""
        ##### Clinical Cause-and-Effect Relationships:
        1. **Hidden Root State (Red Node):**
           - `Sepsis`: The systemic bacterial/viral infection circulating in the bloodstream.
        2. **Observable Bedside Symptoms (Blue Nodes):**
           - `Temperature`: Host immune response causes **Fever** or paradoxical **Hypothermia**.
           - `HeartRate`: Body compensates for low vascular tone with **Tachycardia** (fast heart rate).
           - `BloodPressure_MAP`: Bacterial toxins trigger vasodilation, resulting in severe **Hypotension (Septic Shock)**.
           - `RespirationRate`: Acidosis leads to rapid shallow breathing (**Tachypnea**).
        3. **Why Bayesian Networks?**
           In the ICU, doctors observe the *symptoms* (Blue) to calculate the posterior probability of the *hidden disease* (Red) using **Bayes' Theorem**.
        """)

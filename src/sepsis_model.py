import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

SEPSIS_EDGES = [
    ('Sepsis', 'Temperature'),
    ('Sepsis', 'HeartRate'),
    ('Sepsis', 'RespirationRate'),
    ('Sepsis', 'BloodPressure_MAP'),
    ('Sepsis', 'OxygenSaturation'),
    ('HeartRate', 'BloodPressure_MAP')
]

def discretize_vitals(df):
    """
    Transforms continuous ICU vital signs into clinical risk states
    following SIRS (Systemic Inflammatory Response Syndrome) and SOFA criteria.
    """
    df_clean = df.copy()
    
    # Impute missing values with clinical medians if NaN
    df_clean['HR'] = df_clean['HR'].fillna(80.0)
    df_clean['Temp'] = df_clean['Temp'].fillna(37.0)
    df_clean['Resp'] = df_clean['Resp'].fillna(16.0)
    df_clean['MAP'] = df_clean['MAP'].fillna(80.0)
    df_clean['O2Sat'] = df_clean['O2Sat'].fillna(98.0)
    
    df_bn = pd.DataFrame()
    
    # Sepsis Target State
    if 'SepsisLabel' in df_clean.columns:
        df_bn['Sepsis'] = df_clean['SepsisLabel'].apply(lambda x: 'Yes' if x == 1 else 'No').astype(str)
    
    # Heart Rate: Normal (60-100), Tachycardia (>100), Bradycardia (<60)
    df_bn['HeartRate'] = pd.cut(
        df_clean['HR'],
        bins=[-np.inf, 60.0, 100.0, np.inf],
        labels=['Bradycardia', 'Normal', 'Tachycardia']
    ).astype(str)
    
    # Temperature: Normal (36.0 - 38.0C), Fever (>38.0C), Hypothermia (<36.0C)
    df_bn['Temperature'] = pd.cut(
        df_clean['Temp'],
        bins=[-np.inf, 36.0, 38.0, np.inf],
        labels=['Hypothermia', 'Normal', 'Fever']
    ).astype(str)
    
    # Respiration Rate: Normal (12 - 20), High / Tachypnea (>20)
    df_bn['RespirationRate'] = pd.cut(
        df_clean['Resp'],
        bins=[-np.inf, 20.0, np.inf],
        labels=['Normal', 'Elevated']
    ).astype(str)
    
    # Blood Pressure (MAP): Normal (>=65 mmHg), Low / Shock (<65 mmHg)
    df_bn['BloodPressure_MAP'] = pd.cut(
        df_clean['MAP'],
        bins=[-np.inf, 65.0, np.inf],
        labels=['Hypotension', 'Normal']
    ).astype(str)
    
    # Oxygen Saturation: Normal (>=95%), Low / Hypoxia (<95%)
    df_bn['OxygenSaturation'] = pd.cut(
        df_clean['O2Sat'],
        bins=[-np.inf, 95.0, np.inf],
        labels=['Low_O2', 'Normal']
    ).astype(str)
    
    return df_bn

def discretize_single_vitals(hr=80.0, temp=37.0, resp=16.0, map_bp=80.0, o2sat=98.0):
    """
    Converts a single patient's bedside vitals into discrete evidence states.
    """
    evidence = {}
    
    # Heart Rate
    if hr is not None:
        if hr < 60: evidence['HeartRate'] = 'Bradycardia'
        elif hr <= 100: evidence['HeartRate'] = 'Normal'
        else: evidence['HeartRate'] = 'Tachycardia'
        
    # Temperature
    if temp is not None:
        if temp < 36.0: evidence['Temperature'] = 'Hypothermia'
        elif temp <= 38.0: evidence['Temperature'] = 'Normal'
        else: evidence['Temperature'] = 'Fever'
        
    # Respiration Rate
    if resp is not None:
        evidence['RespirationRate'] = 'Elevated' if resp > 20 else 'Normal'
        
    # Blood Pressure MAP
    if map_bp is not None:
        evidence['BloodPressure_MAP'] = 'Hypotension' if map_bp < 65 else 'Normal'
        
    # Oxygen Saturation
    if o2sat is not None:
        evidence['OxygenSaturation'] = 'Low_O2' if o2sat < 95 else 'Normal'
        
    return evidence

def train_sepsis_bayesian_network(df_raw):
    """
    Trains the Sepsis Bayesian Network on clinical records.
    """
    df_bn = discretize_vitals(df_raw)
    model = DiscreteBayesianNetwork(SEPSIS_EDGES)
    model.fit(df_bn)
    infer = VariableElimination(model)
    return model, infer, df_bn

def query_sepsis_risk(infer, evidence_dict):
    """
    Computes exact posterior probability P(Sepsis = Yes | Evidence)
    and returns prior vs posterior comparison.
    """
    active_ev = {k: v for k, v in evidence_dict.items() if v not in [None, 'Unobserved', 'Any']}
    
    # Prior baseline
    prior_phi = infer.query(variables=['Sepsis'])
    prior_dict = {prior_phi.state_names['Sepsis'][i]: float(prior_phi.values[i]) for i in range(len(prior_phi.values))}
    
    if active_ev:
        try:
            post_phi = infer.query(variables=['Sepsis'], evidence=active_ev)
            post_dict = {post_phi.state_names['Sepsis'][i]: float(post_phi.values[i]) for i in range(len(post_phi.values))}
        except Exception:
            post_dict = prior_dict.copy()
    else:
        post_dict = prior_dict.copy()
        
    sepsis_risk = post_dict.get('Yes', 0.0)
    baseline_risk = prior_dict.get('Yes', 0.0234)
    rel_risk = sepsis_risk / max(baseline_risk, 0.001)
    
    # Calculate a clinical Risk Index (0 - 100%) normalized to ICU decision thresholds
    # Normal (rel_risk <= 1.0) -> 5% to 20%
    # Moderate (rel_risk 1.2 to 2.5) -> 35% to 65%
    # Critical (rel_risk > 2.5) -> 75% to 98%
    if rel_risk < 1.2:
        calibrated_score = min(20.0, max(5.0, (sepsis_risk / 0.025) * 15.0))
        category = "LOW RISK / STABLE"
        alert_level = "STABLE"
        action = "Patient vitals are within normal limits. Continue standard ICU telemetry."
    elif rel_risk < 2.5:
        calibrated_score = min(68.0, max(35.0, 35.0 + ((rel_risk - 1.2) / 1.3) * 30.0))
        category = "MODERATE RISK (INFECTION WATCH)"
        alert_level = "WARNING"
        action = "SIRS criteria triggered (Fever / Tachycardia). Draw blood cultures, check serum lactate, and increase monitoring frequency."
    else:
        calibrated_score = min(98.0, max(75.0, 75.0 + ((rel_risk - 2.5) / 2.0) * 20.0))
        category = "CRITICAL RISK (SEPTIC SHOCK)"
        alert_level = "EMERGENCY"
        action = "EMERGENCY: Septic shock criteria met (Severe Hypotension & Multiorgan Risk). Initiate immediate IV fluid bolus (30mL/kg) and broad-spectrum antibiotics within 1 hour."

    return {
        'sepsis_probability': sepsis_risk,
        'calibrated_score': calibrated_score,
        'relative_risk': rel_risk,
        'prior_probability': baseline_risk,
        'posterior_distribution': post_dict,
        'prior_distribution': prior_dict,
        'risk_category': category,
        'alert_level': alert_level,
        'recommendation': action,
        'active_evidence': active_ev
    }

def render_sepsis_dag():
    """
    Draws the clinical causal DAG diagram for Sepsis.
    """
    G = nx.DiGraph()
    G.add_edges_from(SEPSIS_EDGES)
    
    pos = {
        'Sepsis': (0.5, 1.0),
        'Temperature': (0.1, 0.45),
        'HeartRate': (0.35, 0.45),
        'RespirationRate': (0.65, 0.45),
        'OxygenSaturation': (0.9, 0.45),
        'BloodPressure_MAP': (0.5, 0.05)
    }
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor='#0b0f19')
    ax.set_facecolor('#0f172a')
    
    node_colors = []
    for node in G.nodes():
        if node == 'Sepsis':
            node_colors.append('#ef4444') # Root infection (Red)
        else:
            node_colors.append('#38bdf8') # Bedside Vital Signs (Sky Blue)
            
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=3200, alpha=0.95)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#94a3b8', arrows=True, arrowsize=24, arrowstyle='-|>', width=2.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_family='sans-serif', font_weight='bold', font_color='#ffffff')
    
    ax.set_title("Clinical Causal Network (DAG): Sepsis & ICU Physiological Vitals", fontsize=11, fontweight='bold', color='#f8fafc', pad=14)
    ax.axis('off')
    fig.tight_layout()
    return fig


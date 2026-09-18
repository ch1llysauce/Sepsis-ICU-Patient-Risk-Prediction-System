import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.sepsis_model import (
    train_sepsis_bayesian_network,
    query_sepsis_risk,
    render_sepsis_dag,
    discretize_vitals
)

def main():
    print("=================================================================")
    print("   A Bayesian Network-based Sepsis Risk Detection System         ")
    print("=================================================================")
    
    csv_path = os.path.join("data", "physionet_sepsis_patients.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Run downloader first.")
        return
        
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} ICU hourly records from PhysioNet.")
    
    # Train Bayesian Network
    print("\nTraining Bayesian Network with Maximum Likelihood Estimation...")
    model, infer, df_bn = train_sepsis_bayesian_network(df_raw)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Save bundle
    bundle_path = os.path.join("models", "sepsis_bayesian_bundle.joblib")
    bundle = {
        'model': model,
        'cpds': model.get_cpds(),
        'variables': list(model.nodes()),
        'evidence_nodes': ['HeartRate', 'Temperature', 'RespirationRate', 'BloodPressure_MAP', 'OxygenSaturation']
    }
    joblib.dump(bundle, bundle_path)
    print(f"Saved model bundle -> '{bundle_path}'")
    
    # Save DAG graph
    fig = render_sepsis_dag()
    dag_path = os.path.join("reports", "sepsis_dag.png")
    fig.savefig(dag_path, dpi=300, facecolor='#0b0f19')
    print(f"Saved Clinical DAG Diagram -> '{dag_path}'")
    
    # Quick Test Queries
    print("\n--- Testing Clinical Inference Queries ---")
    
    # Normal Patient
    norm_q = query_sepsis_risk(infer, {
        'HeartRate': 'Normal',
        'Temperature': 'Normal',
        'RespirationRate': 'Normal',
        'BloodPressure_MAP': 'Normal',
        'OxygenSaturation': 'Normal'
    })
    print(f"1. Patient with Normal Vitals   -> Sepsis Risk: {norm_q['sepsis_probability']*100:.1f}% ({norm_q['risk_category']})")
    
    # Critical Septic Patient
    crit_q = query_sepsis_risk(infer, {
        'HeartRate': 'Tachycardia',
        'Temperature': 'Fever',
        'RespirationRate': 'Elevated',
        'BloodPressure_MAP': 'Hypotension',
        'OxygenSaturation': 'Low_O2'
    })
    print(f"2. Patient with Shock Vitals    -> Sepsis Risk: {crit_q['sepsis_probability']*100:.1f}% ({crit_q['risk_category']})")
    
    print("\n=========================================================")
    print("             SEPSIS MODEL TRAINING COMPLETE!             ")
    print("=========================================================")

if __name__ == "__main__":
    main()


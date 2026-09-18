# Sepsis / ICU Patient Risk Prediction System — One-Sitting Build Plan

## Project Overview
**Domain:** Medical Diagnosis — Sepsis Prediction in ICU
**Goal:** Build a working application that estimates a patient's risk of sepsis using a Bayesian Network over clinical monitor readings (Heart Rate, Blood Pressure, Temperature, Respiration Rate, etc.), aiming to flag risk before septic shock occurs.

**Why this is a strong choice:**
- Very easy to explain/pitch: "Ang pasyente sa ICU ay nakakabit sa monitors. Hindi laging agad nakikita kung may sepsis. Gagamit tayo ng Bayesian Network para kalkulahin ang risk bago pa mag-septic shock."
- Publicly available datasets exist (no credentialing wall, unlike MIMIC-III).
- Bayesian Networks applied to ICU monitoring is a classic, well-known textbook example (e.g., the ALARM Network) — very familiar territory for CS/ML instructors, easy to defend/discuss.

## Dataset
- **Name:** PhysioNet/Computing in Cardiology Challenge 2019 — "Early Prediction of Sepsis from Clinical Data"
- **Official source:** https://physionet.org/content/challenge-2019/1.0.0/
- **Access:** Open access, no credentialing/approval needed (unlike MIMIC-III) — direct download.
- **Files to download:** `training_setA.zip` and `training_setB.zip` (each contains per-patient `.psv` files — pipe-separated values, one row per hour of ICU stay)
- **Content per patient file:** Hourly vital signs (HR, Temp, SBP, DBP, MAP, Resp Rate, O2Sat, etc.), lab values, demographics, and a `SepsisLabel` column (1 = sepsis onset, 0 = no sepsis) — this is your ground truth label.
- **Alt/simpler option if setA/B feels too heavy:** Build a smaller synthetic clinical dataset (simulate HR, BP, Temp, RR distributions for septic vs non-septic patients based on published clinical thresholds) — useful as a fallback or for a clean demo case, but real PhysioNet data is stronger for credibility.

## Requirements Mapping
1. **Working application/program** → Full pipeline app (patient vitals input → Bayesian inference → risk output)
2. **Publicly available dataset** → PhysioNet Challenge 2019 Sepsis dataset (open access)

## System Features to Implement

### 1. Data Ingestion Module
- Load `.psv` files (per-patient hourly records) using `pandas.read_csv(sep='|')`
- Combine/sample a manageable subset of patients (don't need all thousands of patients for a one-sitting demo — sample a few hundred)

### 2. Data Preprocessing Module
- Handle missing values (common in this dataset — forward-fill or use only rows with key vitals present)
- Discretize continuous vitals into clinical risk bins (this is required for a classic discrete Bayesian Network), e.g.:
  - Heart Rate: Normal (60–100) / Tachycardia (>100) / Bradycardia (<60)
  - Temperature: Normal (36.5–37.5°C) / Fever (>38°C) / Hypothermia (<36°C)
  - Respiration Rate: Normal (12–20) / Elevated (>20)
  - Blood Pressure (MAP): Normal (>65 mmHg) / Low/Hypotension (<65 mmHg)
- Map `SepsisLabel` as the target node (Sepsis: Yes/No)

### 3. Bayesian Network Construction
- Define network structure (nodes + edges), e.g.:
  - `HeartRate → Sepsis`
  - `Temperature → Sepsis`
  - `RespirationRate → Sepsis`
  - `BloodPressure → Sepsis`
  - (Optionally chain some vitals together if clinically justified, similar to ALARM network style)
- Use `pgmpy` (Python library) to build the Bayesian Network structure
- Learn Conditional Probability Tables (CPTs) from the dataset (Maximum Likelihood Estimation via `pgmpy`)

### 4. Inference Engine
- Given new patient vitals (discretized), run inference (`pgmpy.inference.VariableElimination`) to compute `P(Sepsis = Yes | evidence)`
- Output a risk probability/score, not just binary — this is the key value of Bayesian approach (uncertainty-aware risk, not just classification)

### 5. Evaluation
- Split dataset into train/test
- Evaluate against ground truth `SepsisLabel`: accuracy, precision/recall, AUC
- Show a few example patients with high vs low computed risk

### 6. Prediction Interface (the "application" part)
- Simple **Streamlit** web app:
  - Input fields/sliders for HR, Temp, BP, Respiration Rate (simulate "monitor readings")
  - Button: "Compute Sepsis Risk"
  - Output: Risk probability (%) + risk level label (Low/Medium/High) + simple explanation of which vitals are driving the risk
  - Optional: visualize the Bayesian Network graph itself (nodes + edges) for presentation value

### 7. Logging/Reporting Feature
- Save each risk assessment (timestamp, input vitals, computed risk) to a local CSV
- Useful for demoing "patient monitoring over time" narrative

## Suggested Build Order (One Sitting)
1. Download PhysioNet Challenge 2019 dataset (`training_setA.zip`) (15–20 min)
2. Load and explore a sample of patient files, understand columns/missingness (20–30 min)
3. Preprocess: discretize vitals into risk bins, build combined dataset with SepsisLabel (45 min)
4. Build Bayesian Network structure + learn CPTs using `pgmpy` (45–60 min)
5. Build inference function (given vitals → risk probability) (20–30 min)
6. Evaluate against test set (20 min)
7. Build Streamlit interface for live risk prediction (45–60 min)
8. Add logging/history feature (15–20 min)
9. Test end-to-end, polish UI/labels, prepare explanation of network structure for defense (20–30 min)

## Tech Stack
- Python
- `pandas`, `numpy` — data handling
- `pgmpy` — Bayesian Network construction, CPT learning, inference
- `matplotlib`/`networkx` — visualize the Bayesian Network graph
- `streamlit` — web app interface
- `scikit-learn` — for train/test split and evaluation metrics only

## Stretch Goals (if time remains)
- Add more nodes to the network (e.g., O2 Saturation, Age, WBC count if available) for a richer ALARM-style network
- Add a simple time-trend feature (e.g., "vitals worsening over last 3 hours") to simulate early-warning behavior
- Deploy Streamlit app publicly (Streamlit Community Cloud)
- Compare Bayesian Network performance vs. a simple Logistic Regression baseline, to argue for interpretability/uncertainty benefits of the Bayesian approach

## Key Talking Points for Defense/Presentation
- Bayesian Networks handle uncertainty and missing data naturally — very relevant in real ICU settings where not all readings are always available.
- The structure of the network (which vitals influence sepsis) is interpretable and can be explained to clinicians, unlike black-box models.
- This mirrors the same reasoning approach as the classic ALARM Network (patient monitoring Bayesian Network) taught in AI/ML courses — grounds the project in established theory.
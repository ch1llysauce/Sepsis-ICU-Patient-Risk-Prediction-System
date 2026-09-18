import os
import io
import urllib.request
import concurrent.futures
import pandas as pd

def download_patient(patient_id):
    url = f"https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/p{patient_id:06d}.psv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()), sep='|')
            df['PatientID'] = f"p{patient_id:06d}"
            return df
    except Exception:
        return None

def download_physionet_sepsis_dataset(n_patients=300, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "physionet_sepsis_patients.csv")
    
    print(f"Downloading {n_patients} patient records from PhysioNet Challenge 2019...")
    
    patient_ids = list(range(1, n_patients + 1))
    dfs = []
    
    # 25 concurrent threads for fast parallel download
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        results = executor.map(download_patient, patient_ids)
        for i, df in enumerate(results, 1):
            if df is not None and len(df) > 0:
                dfs.append(df)
            if i % 50 == 0 or i == n_patients:
                print(f"  Progress: {i}/{n_patients} patients fetched...")

    if not dfs:
        raise RuntimeError("Failed to download patient data from PhysioNet.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Success! Downloaded {len(dfs)} patients with {len(combined)} hourly ICU records.")
    print(f"Dataset saved to: {out_csv}")
    print("Sepsis Label Distribution:")
    print(combined['SepsisLabel'].value_counts())
    return combined

if __name__ == "__main__":
    download_physionet_sepsis_dataset(n_patients=300)


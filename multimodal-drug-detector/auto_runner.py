import os
import requests
import json
import time
from datetime import datetime

# API Endpoints
PREDICTION_API = "http://127.0.0.1:9000/analyze_patient"  # Using EEP Server now
EXPLANATION_API = "http://127.0.0.1:9000/analyze_with_explanation"
DATA_DIR = r"C:\Users\yourpath\data_to_predict"
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
WAIT_ENABLED = True
WAIT_SECONDS = 86400  # 24 hours

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_patient(folder_path):
    mri_path = os.path.join(folder_path, "mri.nii.gz")
    ehr_path = os.path.join(folder_path, "ehr.json")
    result_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(folder_path)}_result.json")
    heatmap_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(folder_path)}_heatmap.png")

    if not os.path.exists(mri_path) or not os.path.exists(ehr_path):
        print(f"[!] Missing files in {folder_path}")
        return False

    with open(ehr_path, 'r') as f:
        ehr_data = json.load(f)

    # Standard prediction
    try:
        with open(mri_path, 'rb') as mri_file:
            files = {'file': mri_file}
            data = {'EHR_features': json.dumps(ehr_data)}
            
            # Get both regular prediction and heatmap
            pred_response = requests.post(PREDICTION_API, files=files, data=data)
            pred_response.raise_for_status()
            result = pred_response.json()
            
            # Save basic prediction result
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            # If drug abuse detected, get heatmap
            if result.get('prediction', {}).get('class', 0) == 1:
                mri_file.seek(0)  # Rewind file for second request
                heatmap_response = requests.post(EXPLANATION_API, files=files, data=data)
                
                if heatmap_response.status_code == 200:
                    with open(heatmap_path, 'wb') as f:
                        f.write(heatmap_response.content)
                    print(f"[🌋] Heatmap saved to {heatmap_path}")
                else:
                    print(f"[!] Heatmap generation failed: {heatmap_response.text}")
            
            return True

    except Exception as e:
        print(f"[X] Error processing {folder_path}: {str(e)}")
        return False

def run_batch():
    print(f"\n[{datetime.now()}] Starting processing batch...")
    ensure_directory(OUTPUT_DIR)
    
    processed = 0
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            if process_patient(folder_path):
                processed += 1
    
    print(f"[✅] Batch complete. Processed {processed} patients.")
    return processed > 0

if __name__ == "__main__":
    while True:
        if run_batch() and WAIT_ENABLED:
            print(f"[⏳] Next batch in {WAIT_SECONDS/3600} hours...")
            time.sleep(WAIT_SECONDS)
        else:
            print("[🛑] No patients processed or WAIT_ENABLED=False. Exiting.")
            break

import time
import os
import requests
import json

API_URL = "http://127.0.0.1:5000/drug_abuse_detector"  # Adjust if running remotely
DATA_DIR = "data_to_predict"  # Folder structure: data_to_predict/patient123/{mri.nii, ehr.json}

def run_predictions():
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        mri_path = os.path.join(folder_path, "mri.nii.gz")
        ehr_path = os.path.join(folder_path, "ehr.json")
        verdict_path = os.path.join(folder_path, "verdict.json")

        # 🔁 Skip folders already processed
        if os.path.exists(verdict_path):
            print(f"[⏭️] Skipping {folder}, already processed.")
            continue

        if not os.path.exists(mri_path) or not os.path.exists(ehr_path):
            print(f"[!] Missing MRI or EHR file for {folder}")
            continue

        with open(ehr_path, 'r') as f:
            ehr_data = json.load(f)

        files = {
            'file': open(mri_path, 'rb'),
        }
        data = {
            'EHR_features': json.dumps(ehr_data)
        }

        try:
            response = requests.post(API_URL, files=files, data=data)
            result = response.json()
            print(f"[✓] Prediction for {folder}: {result}")

            # ✅ Save result to skip next time
            with open(verdict_path, 'w') as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            print(f"[X] Error for {folder}: {e}")

if __name__ == "__main__":
    while True:
        print("[🔁] Running prediction cycle...")
        run_predictions()
        print("[⏳] Sleeping for 12 hours...\n")
        time.sleep(86400)  

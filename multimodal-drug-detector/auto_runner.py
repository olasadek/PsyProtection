import os
import requests
import json
import time

API_URL = "http://127.0.0.1:5000/drug_abuse_detector"
RAG_URL = "http://127.0.0.1:8000/ask_question"
DATA_DIR = r"C:\Users\Dell\Downloads\predictions\data_to_predict"
WAIT_ENABLED = True  # Set to False to skip the 12-hour wait during testing

def run_predictions():
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        mri_path = os.path.join(folder_path, "mri.nii.gz")
        ehr_path = os.path.join(folder_path, "ehr.json")
        verdict_path = os.path.join(folder_path, "verdict.json")

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

            predicted_class = result["prediction"].get("class", 0)

            if predicted_class == 1:
                medication = ehr_data.get("medication", "psychiatric medication")
                illness = ehr_data.get("illness", "mental illness")
                rag_query = f"newest treatments for {medication} addiction for {illness} patients"
                print(f"[🔎] Suggested RAG query: {rag_query}")
                result["prediction"]["suggested_query"] = rag_query

                try:
                    rag_response = requests.post(RAG_URL, json={"query": rag_query})
                    rag_answer = rag_response.json().get("answer", "No answer returned.")
                    print(f"[📚] RAG answer: {rag_answer}")
                    result["prediction"]["rag_answer"] = rag_answer
                except Exception as e:
                    print(f"[X] Error calling RAG: {e}")
                    result["prediction"]["rag_answer"] = "Error retrieving RAG answer."

            else:
                print("✅ Patient looks good (not a drug abuser).")

            with open(verdict_path, 'w') as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            print(f"[X] Error for {folder}: {e}")

if __name__ == "__main__":
    while True:
        print("[🔁] Running prediction cycle...")
        run_predictions()
        print("✅ All patients processed.")
        if WAIT_ENABLED:
            print("[⏳] Waiting 24 hours before next batch...\n")
            time.sleep(86400)  # if you want to test it quickly, put it on 10 seconds 

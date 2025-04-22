import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
import tempfile
import torchvision.transforms as transforms
import joblib
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder

# Prepare cache for model files
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_if_missing(file_id, filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

# File downloads
MODEL_PATH   = download_if_missing("1hrGd-m641Imceu86vLljXPlQWa7aXO00", "multi_modal_diagnostic_model.pkl")
SCALER_PATH  = download_if_missing("1dUjpSyq3eXzP00QThmn94InR5XTO6XSJ", "scaler.pkl")
ENCODER_PATH = download_if_missing("1eJ96Wp1d8M4Q1DWpzM0xLNc7-TOGGISP", "encoder.pkl")
DATABASE_PATH = download_if_missing("1-FbV8JixOrZdzU_r8HAxFFymaAlB7VXk", "predictions.db")

# App setup
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Load model + preprocessors
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

FRIENDLY_TO_MODEL_KEYS = {
    "age": "age", "years of education": "educ_yr", "occupation level": "occup",
    "annual income": "income", "handedness": "laterality", "AMAI category": "amai",
    "AMAI score": "amai_score", "years since first drug use": "years.begin",
    "age of drug onset": "drug.age.onset", "days since last drug use": "days.last.use",
    "weekly dose": "week.dose", "tobacco consumed since last year": "tobc.lastyear",
    "tobacco per day": "tobc.day", "years of tobacco use": "tobc.totyears",
    "current illness": "illness", "current medication": "medication"
}

def validate_mri_file(file_obj):
    try:
        magic = file_obj.read(4)
        file_obj.seek(0)
        if magic not in [b'\x6E\x69\x31\x00', b'n+1']:
            return False, "Invalid file header (not NIfTI)"
        img = nib.load(file_obj)
        file_obj.seek(0)
        if len(img.shape) not in [3, 4]:
            return False, f"Invalid dimensions {img.shape}. Expected 3D or 4D"
        data = img.get_fdata()
        if np.any(np.isnan(data)):
            return False, "MRI contains NaN values"
        return True, "Valid MRI file"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      patient_id TEXT,
                      prediction_prob REAL,
                      prediction_class INTEGER,
                      verdict TEXT,
                      ehr_data TEXT,
                      explanation_url TEXT,
                      query_answer TEXT,
                      timestamp DATETIME,
                      heatmap_path TEXT,
                      validation_status TEXT)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON predictions(patient_id)')
        conn.commit()

def store_prediction(patient_id, prediction_prob, prediction_class, verdict, 
                    ehr_data, explanation_url, query_answer, heatmap_path, validation_status):
    with sqlite3.connect(DATABASE_PATH) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (patient_id, prediction_prob, prediction_class, verdict,
                      ehr_data, explanation_url, query_answer, timestamp, 
                      heatmap_path, validation_status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (patient_id, prediction_prob, prediction_class, verdict,
                   json.dumps(ehr_data), explanation_url, query_answer, 
                   datetime.now(), heatmap_path, validation_status))
        conn.commit()

@app.route('/drug_abuse_detector', methods=['POST'])
def drug_abuse_detector():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No MRI file provided"}), 400
        if 'EHR_features' not in request.form:
            return jsonify({"error": "No EHR features provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected MRI file"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            file.save(temp_file.name)
            try:
                mri_scan = nib.load(temp_file.name)
                mri_data = mri_scan.get_fdata()
            except Exception as e:
                return jsonify({"error": f"Invalid MRI file: {str(e)}"}), 400

        ehr_dict = json.loads(request.form['EHR_features'])
        ehr_model_ready = {FRIENDLY_TO_MODEL_KEYS.get(k, k): v for k, v in ehr_dict.items()}
        for key, value in ehr_model_ready.items():
            if isinstance(value, str):
                if value.replace('.', '', 1).isdigit():
                    ehr_model_ready[key] = float(value) if '.' in value else int(value)

        ehr_df = pd.DataFrame([ehr_model_ready])
        numeric_cols = ehr_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = ehr_df.select_dtypes(include=['object', 'category']).columns
        ehr_df[numeric_cols] = scaler.transform(ehr_df[numeric_cols])
        ehr_df[categorical_cols] = encoder.transform(ehr_df[categorical_cols])
        ehr_tensor = torch.from_numpy(ehr_df.values).float()

        middle_slice = mri_data[:, :, mri_data.shape[2] // 2]
        middle_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice))
        mri_tensor = transform(middle_slice).unsqueeze(0).float()

        with torch.no_grad():
            predictions = model(mri_tensor, ehr_tensor)
            predicted_prob = predictions.item()
            predicted_class = int(predictions >= 0.5)

        verdict = "Drug abuser" if predicted_class == 1 else "Not a drug abuser"
        response_data = {
            "prediction": {
                "probability": float(predicted_prob),
                "class": predicted_class,
                "description": verdict
            },
            "ehr_data": ehr_dict,
            "patient_id": request.form.get('patient_id', 'unknown')
        }

        store_prediction(
            patient_id=response_data["patient_id"],
            prediction_prob=float(predicted_prob),
            prediction_class=predicted_class,
            verdict=verdict,
            ehr_data=ehr_dict,
            explanation_url="",
            query_answer="",
            heatmap_path="",
            validation_status="valid"
        )

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        patient_id = request.args.get('patient_id')
        limit = request.args.get('limit', 100)

        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            if patient_id:
                c.execute('SELECT * FROM predictions WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?', 
                         (patient_id, limit))
            else:
                c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (limit,))

            predictions = [dict(row) for row in c.fetchall()]

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)

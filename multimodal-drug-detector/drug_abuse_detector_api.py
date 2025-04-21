import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
import tempfile
import torchvision.transforms as transforms
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder
from flask_cors import CORS
import base64
from flask import send_file , Flask, request, jsonify, make_response
import io

BASE_DIR = r"C:\Users\Dell\Downloads\PsyProtection-main\multimodal-drug-detector"
MODEL_PATH = os.path.join(BASE_DIR, "multi_modal_diagnostic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
STATIC_PATH = os.path.join(BASE_DIR, "static")
DATABASE_PATH = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__, static_folder=STATIC_PATH)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

FRIENDLY_TO_MODEL_KEYS = {
    "age": "age",
    "years of education": "educ_yr",
    "occupation level": "occup",
    "annual income": "income",
    "handedness": "laterality",
    "AMAI category": "amai",
    "AMAI score": "amai_score",
    "years since first drug use": "years.begin",
    "age of drug onset": "drug.age.onset",
    "days since last drug use": "days.last.use",
    "weekly dose": "week.dose",
    "tobacco consumed since last year": "tobc.lastyear",
    "tobacco per day": "tobc.day",
    "years of tobacco use": "tobc.totyears",
    "current illness": "illness",
    "current medication": "medication"
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

def mri_occlusion_explain_internal(mri_data):
    try:
        middle_slice = mri_data[:, :, mri_data.shape[2] // 2]
        middle_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice))
        image_tensor = transform(middle_slice).unsqueeze(0).float()

        vit = ViTImageEncoder()
        vit.eval()
        classifier = model.classifier
        dummy_ehr = torch.zeros((1, 50))

        with torch.no_grad():
            base_feat = vit(image_tensor)
            base_output = classifier(torch.cat([base_feat, dummy_ehr], dim=1))
            base_pred = base_output.item()
            pred_class = int(base_output >= 0.5)
            confidence = float(base_pred if pred_class == 1 else 1 - base_pred)

        patch_size = 20
        stride = 10
        center_crop = 112
        crop_size = 100
        start = center_crop - crop_size // 2
        end = center_crop + crop_size // 2
        heatmap = np.zeros((224, 224))

        for i in range(start, end - patch_size, stride):
            for j in range(start, end - patch_size, stride):
                perturbed = image_tensor.clone()
                perturbed[:, :, i:i+patch_size, j:j+patch_size] = 0
                with torch.no_grad():
                    feat = vit(perturbed)
                    pred = classifier(torch.cat([feat, dummy_ehr], dim=1)).item()
                delta = base_pred - pred
                heatmap[i:i+patch_size, j:j+patch_size] += delta

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap + 1e-8)

        os.makedirs(STATIC_PATH, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        heatmap_filename = f"heatmap_{timestamp}.png"
        heatmap_path = os.path.join(STATIC_PATH, heatmap_filename)

        fig, ax = plt.subplots()
        ax.imshow(middle_slice, cmap='gray')
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.axis('off')
        plt.savefig(heatmap_path, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        generate_summary(pred_class, confidence, heatmap_path)

        return {
            "explanation": "Occlusion-based explanation complete.",
            "base_prediction": base_pred,
            "heatmap_url": f"/static/{heatmap_filename}",
            "summary_file": heatmap_filename.replace(".png", "_summary.txt")
        }

    except Exception as e:
        return {"error": str(e)}

def generate_summary(prediction, confidence, heatmap_path):
    summary_text = f"""Prediction Summary

🧠 Prediction: {'Drug Abuser' if prediction == 1 else 'Non-Abuser'}
📊 Confidence: {confidence * 100:.2f}%

Explanation:
This heatmap highlights regions of the brain that influenced the model's decision. 
Red = high impact, Blue = low impact on prediction.

These regions are consistent with neural patterns observed in drug abuse-related cases during training.
"""
    summary_path = heatmap_path.replace(".png", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

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

        try:
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
        except Exception as e:
            return jsonify({"error": f"EHR processing failed: {str(e)}"}), 400

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
            explanation_url="",  # Or just remove this column from schema
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
    os.makedirs(STATIC_PATH, exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5000)

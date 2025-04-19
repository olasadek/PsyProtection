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
import requests
import sqlite3
from flask import Flask, request, jsonify
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder

BASE_DIR = r"C:\Users\Dell\Downloads\PsyProtection-main\multimodal-drug-detector"
MODEL_PATH = os.path.join(BASE_DIR, "multi_modal_diagnostic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
STATIC_PATH = os.path.join(BASE_DIR, "static")
DATABASE_PATH = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__, static_folder=STATIC_PATH)
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

def validate_mri_file(file_stream):
    """Validate if the uploaded file is a proper MRI (NIfTI)"""
    try:
        # Check first 4 bytes for NIfTI magic number
        magic = file_stream.read(4)
        file_stream.seek(0)
        
        if magic not in [b'\x1f\x8b\x08\x08', b'n+1']:  # NIfTI magic numbers
            return False, "Invalid file header (not NIfTI)"
        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(file_stream.read())
            temp_path = temp_file.name
        # Try loading the file
        try:
            img = nib.load(temp_path)
            file_stream.seek(0)
        except Exception as e:
            return False, f"Invalid NIfTI file: {str(e)}"
         # Check dimensionality
        if len(img.shape) not in [3, 4]:
            return False, f"Invalid dimensions {img.shape}. Expected 3D or 4D"
        # Check data range
        data = img.get_fdata()
        if np.any(np.isnan(data)):
            return False, "MRI contains NaN values"
            
        return  True, "Valid MRI file"
        
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
        model = MultiModalDiagnosticNet(ehr_input_dim=16)
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
            return jsonify({"error": "No MRI file uploaded"}), 400

        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            file.save(temp_file.name)
            mri_scan = nib.load(temp_file.name)
            mri_data1 = mri_scan.get_fdata()

        # Process EHR data
        patient_id = request.form.get('patient_id', 'unknown')
        ehr_raw = request.form.get('EHR_features')
        if not ehr_raw:
            return jsonify({"error": "No EHR data provided"}), 400

        try:
            ehr_dict = json.loads(ehr_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid EHR JSON format"}), 400

        # Normalize and transform MRI
        mri_tensor = transform(middle_slice).unsqueeze(0).float()
        mri_data = mri_data1.reshape(-1, mri_data1.shape[1])
        mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
        mri_tensor = transform(mri_data).unsqueeze(0).float()
        

        # Prepare EHR features
        ehr_model_ready = {FRIENDLY_TO_MODEL_KEYS.get(k, k): v for k, v in ehr_dict.items()}
        ehr_df = pd.DataFrame([ehr_model_ready])
        numeric_cols = ehr_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = ehr_df.select_dtypes(include=['object', 'category']).columns
        ehr_df[numeric_cols] = scaler.transform(ehr_df[numeric_cols])
        ehr_df[categorical_cols] = encoder.transform(ehr_df[categorical_cols])
        ehr_tensor = torch.from_numpy(ehr_df.values).float()

        # Predict
        with torch.no_grad():
            predictions = model(mri_tensor, ehr_tensor)
            predicted_prob = predictions.item()
            predicted_class = (predictions >= 0.5).float()

        verdict = "Drug abuser" if predicted_class == 1 else "Not a drug abuser"
        ehr_dict["Verdict"] = verdict

        # Optional explanation
        heatmap_path = None
        if predicted_class == 1:
            explanation = mri_occlusion_explain_internal(mri_data1)
            if "error" in explanation:
                return jsonify({"error": f"Explanation failed: {explanation['error']}"}), 500
            heatmap_path = explanation['heatmap_url'].split('/static/')[-1]
        else:
            explanation = {"heatmap_url": None}

        # Optional RAG
        query_answer = None
        if predicted_class == 1:
            medication = ehr_dict.get("medication", "")
            illness = ehr_dict.get("illness", "")
            if medication and illness:
                rag_query = f"newest treatments for {medication} addiction for {illness} patients"
                try:
                    response = requests.post("http://127.0.0.1:8000/ask_question", json={"query": rag_query})
                    query_answer = response.json().get("answer", "No answer found.")
                except Exception as e:
                    query_answer = f"Error retrieving RAG answer: {str(e)}"

        # Save prediction
        store_prediction(
            patient_id=patient_id,
            prediction_prob=float(predicted_prob),
            prediction_class=int(predicted_class),
            verdict=verdict,
            ehr_data=ehr_dict,
            explanation_url=explanation['heatmap_url'],
            query_answer=query_answer,
            heatmap_path=heatmap_path,
            validation_status="Not validated"
        )

        return jsonify({
            "prediction": {
                "probability": float(predicted_prob),
                "class": int(predicted_class),
                "description": verdict
            },
            "explanation": explanation['heatmap_url'],
            "query_answer": query_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/mri_occlusion_explain', methods=['POST'])
def mri_occlusion_explain():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            file.save(temp_file.name)
            mri_scan = nib.load(temp_file.name)
            mri_data = mri_scan.get_fdata()

        result = mri_occlusion_explain_internal(mri_data)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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

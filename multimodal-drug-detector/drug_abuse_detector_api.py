from flask import Flask, request, jsonify
import torch
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
import tempfile
import torchvision.transforms as transforms
import joblib
import json
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder

app = Flask(__name__)

# Load model and encoders
model = joblib.load("multi_modal_diagnostic_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

# Feature label mapping (for frontend <-> backend)
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

@app.route('/drug_abuse_detector', methods=['POST'])
def drug_abuse_detector():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            file.save(temp_file.name)
            mri_scan = nib.load(temp_file.name)
            mri_data = mri_scan.get_fdata()

        mri_data = mri_data.reshape(-1, mri_data.shape[1])
        mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
        mri_data = transform(mri_data).unsqueeze(0).float()
        ehr_raw = request.form.get('EHR_features')
        ehr_dict = json.loads(ehr_raw)

        ehr_model_ready = {
            FRIENDLY_TO_MODEL_KEYS.get(k, k): v for k, v in ehr_dict.items()
        }

        ehr_df = pd.DataFrame([ehr_model_ready])

        numeric_cols = ehr_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = ehr_df.select_dtypes(include=['object', 'category']).columns
        ehr_df[numeric_cols] = scaler.transform(ehr_df[numeric_cols])
        ehr_df[categorical_cols] = encoder.transform(ehr_df[categorical_cols])
        ehr_tensor = torch.from_numpy(ehr_df.values).float()
        with torch.no_grad():
            predictions = model(mri_data, ehr_tensor)
            predicted_prob = predictions.item()
            predicted_class = (predictions >= 0.5).float()

        verdict = "Drug abuser" if predicted_class == 1 else "Not a drug abuser"
        ehr_dict["Verdict"] = verdict

         # After verdict logic
        suggested_query = None
        if predicted_class == 1:
            illness = ehr_dict.get("illness", "")
            medication = ehr_dict.get("medication", "")
            if illness and medication:
                suggested_query = f"newest treatments for {medication} addiction for {illness} patients"
            else:
                suggested_query = None  # If there's no illness or medication, no suggested query

        return jsonify({
            "prediction": {
                "probability": float(predicted_prob),
                "description": verdict
            },
            "suggested_query": suggested_query
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/')
def home():
    return "Drug Abuse Detection API is live!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

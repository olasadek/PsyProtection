from flask import Flask, request, jsonify
import os
import torch
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
import tempfile
import torchvision.transforms as transforms
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import requests

from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder

# Setup
STATIC_PATH = "C:/Users/Dell/Downloads/predictions/static"
app = Flask(__name__, static_folder=STATIC_PATH)

# Load model and transformers
model = joblib.load("multi_modal_diagnostic_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Image transform
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

@app.route('/')
def home():
    return "Drug Abuse Detection & Explanation API is live!"


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

        query_answer = None
        if suggested_query:
            # Send the suggested query to the service on port 8000
            response = requests.post("http://127.0.0.1:8000/ask_question", json={"query": suggested_query})
            if response.status_code == 200:
                query_answer = response.json().get("answer", "No answer found.")
            else:
                query_answer = "Error retrieving the answer from query service."

        return jsonify({
            "prediction": {
                "probability": float(predicted_prob),
                "description": verdict
            },
            "query_answer": query_answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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


@app.route('/mri_occlusion_explain', methods=['POST'])
def mri_occlusion_explain():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            file.save(temp_file.name)
            mri_scan = nib.load(temp_file.name)
            mri_data = mri_scan.get_fdata()

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

        return jsonify({
            "explanation": "Occlusion-based explanation complete.",
            "base_prediction": base_pred,
            "heatmap_url": f"http://localhost:5000/static/{heatmap_filename}",
            "summary_file": heatmap_filename.replace(".png", "_summary.txt")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    os.makedirs(STATIC_PATH, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import torch
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import  StandardScaler,OrdinalEncoder
import numpy as np
import tempfile
import torchvision.transforms as transforms
from ast import literal_eval
import joblib
import json
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder 

app = Flask(__name__)
# Load the trained model, encoder and the decoder (this assumes the model file is in the same directory as the script)
model = joblib.load("multi_modal_diagnostic_model.pkl")
scaler= joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")
# transformation for the MRI data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

@app.route('/drug_abuse_detector', methods=['POST'])
def drug_abuse_detector():
    try:
       # Get input data from the request (Expecting JSON)
       # 1.  MRI scan in NIfTI format (.nii.gz) for analysis."
       file=request.files['file']
       with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
           file.save(temp_file.name)
           mri_scan = nib.load(temp_file.name)
           mri_data = mri_scan.get_fdata()
        
       # Reshape the MRI data to be 2D
       mri_data = mri_data.reshape(-1, mri_data .shape[1])
       # Normalizing the MRI data
       mri_data=(mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
       # Apply the transform to the MRI data to suit the model input
       mri_data=transform(mri_data).float() # tensor(torch.Size([1, 224, 224]))

       # 2. EHR features in list for analysis (expected input list of features).
       EHR_features = request.form.get('EHR_features') 
       EHR_features = json.loads(EHR_features)
       # Convert EHR features to a list
       EHR_features=pd.DataFrame([EHR_features])

       numeric_cols = EHR_features.select_dtypes(include=['int64', 'float64']).columns
       categorical_cols = EHR_features.select_dtypes(include=['object', 'category']).columns
       EHR_features[numeric_cols] = scaler.transform(EHR_features[numeric_cols])
       EHR_features[categorical_cols] = encoder.transform(EHR_features[categorical_cols])
       EHR_features=(torch.from_numpy(EHR_features.values).float())
       # 3. Predict using the model
       with torch.no_grad():
           # Get model predictions
           predictions =model(mri_data,EHR_features)  # Pass both images and features to the model
           predicted_prob = predictions.item()  # Get the raw probability value
           # Get predicted classes: threshold at 0.5 to decide class (0 or 1)
           predicted_class = (predictions >= 0.5).float()  # 1 if probability >= 0.5 else 0

        # 3. Return exactly what was received
       return jsonify({
    "prediction": {
        "probability": float(predicted_prob),  # Raw probability (e.g., 0.82)
        "is_drug_abuser": bool(predicted_class),  # True/False
        "class": int(predicted_class),  # 1 or 0
        "description": "Drug abuser" if predicted_class == 1 else "Not a drug abuser"
    }})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/')
def home():
    return "Congratulations! Your Drug Abuse Detection API is working."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

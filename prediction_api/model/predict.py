import torch
import nibabel as nib
import numpy as np
import tempfile
import torchvision.transforms as transforms
import joblib
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder 

# Load the trained model
model = joblib.load("multi_modal_diagnostic_model.pkl")

# Transformation for MRI
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

def predict_mri_and_ehr(mri_bytes, ehr_features_list):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
        temp_file.write(mri_bytes)
        mri_scan = nib.load(temp_file.name)
        mri_data = mri_scan.get_fdata()

    mri_data = mri_data.reshape(-1, mri_data.shape[1])
    mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
    mri_data = transform(mri_data).float()

    ehr_tensor = torch.tensor(ehr_features_list, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predictions = model(mri_data, ehr_tensor)
        predicted_prob = predictions.item()
        predicted_class = float(predictions >= 0.5)

    return {
        "probability": round(predicted_prob, 4),
        "is_drug_abuser": bool(predicted_class),
        "class": int(predicted_class),
        "description": "Drug abuser" if predicted_class == 1 else "Not a drug abuser"
    }

"""
API Endpoint Documentation for Drug Abuse Detection System

Base URLs:
- Main API: http://127.0.0.1:5000
- RAG API: http://127.0.0.1:8000

Note: Replace base URLs with production endpoints when deployed.
"""

# MRI VALIDATION 
"""
POST /validate_mri
Purpose: Validate an MRI file before full processing

Request Format (multipart/form-data):
- file: MRI scan file (.nii.gz)

Response:
{
    "valid": boolean,
    "message": string,
    "filename": string
}

Example Usage (JavaScript):
const validateMRI = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('http://127.0.0.1:5000/validate_mri', {
        method: 'POST',
        body: formData
    });
    return await response.json();
};
"""

#  MAIN PREDICTION 
"""
POST /drug_abuse_detector
Purpose: Submit MRI and EHR data for drug abuse prediction

Request Format (multipart/form-data):
- file: MRI scan file (.nii.gz)
- patient_id: string (optional)
- EHR_features: JSON string of patient data

Required EHR Fields:
{
    "age": number,
    "educ_yr": number,
    "occup": string,
    "income": number,
    "laterality": string,
    "amai": string,
    "amai_score": number,
    "years.begin": number,
    "drug.age.onset": number,
    "days.last.use": number,
    "week.dose": number,
    "tobc.lastyear": number,
    "tobc.day": number,
    "tobc.totyears": number,
    "illness": string,
    "medication": string
}

Response:
{
    "prediction": {
        "probability": number (0-1),
        "class": number (0 or 1),
        "description": string
    },
    "validation": string,
    "explanation": string (URL to heatmap if class=1),
    "query_answer": string (treatment info if class=1)
}

Example Usage (JavaScript):
const predict = async (mriFile, ehrData) => {
    const formData = new FormData();
    formData.append('file', mriFile);
    formData.append('EHR_features', JSON.stringify(ehrData));
    const response = await fetch('http://127.0.0.1:5000/drug_abuse_detector', {
        method: 'POST',
        body: formData
    });
    return await response.json();
};
"""

# PREDICTION HISTORY 
"""
GET /get_predictions
Purpose: Retrieve past predictions

Query Parameters:
- patient_id: string (optional)
- limit: number (default: 100)

Response:
[
    {
        "id": number,
        "patient_id": string,
        "prediction_prob": number,
        "prediction_class": number,
        "verdict": string,
        "timestamp": string (ISO format),
        "heatmap_path": string,
        "explanation_url": string
    }
]

Example Usage (JavaScript):
const getHistory = async (patientId) => {
    const response = await fetch(
        `http://127.0.0.1:5000/get_predictions?patient_id=${patientId}`
    );
    return await response.json();
};
"""

# RAG SYSTEM 
"""
POST /ask_question
Purpose: Query the treatment information system

Request Format (JSON):
{
    "query": string
}

Response:
{
    "answer": string
}

Example Usage (JavaScript):
const askQuestion = async (question) => {
    const response = await fetch('http://127.0.0.1:8000/ask_question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question })
    });
    return await response.json();
};
"""

#  AUTO-RUNNER CONFIG 
"""
The auto-runner system monitors the data directory at:
C:\Users\Dell\Downloads\predictions\data_to_predict

Folder Structure:
- patient_1/
  - mri.nii.gz
  - ehr.json
  - verdict.json (created after processing)

The system automatically:
1. Processes unprocessed patient folders
2. Calls the prediction API
3. For positive cases, queries the RAG system
4. Stores results in verdict.json
5. Runs every 24 hours (configurable)
"""

# ====================== ERROR HANDLING ======================
"""
All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid input)
- 500: Server error

Error responses include:
{
    "error": string,
    "details": string (optional)
}
"""

#  EXAMPLE EHR PAYLOAD 
EXAMPLE_EHR = {
    "age": 35,
    "educ_yr": 12,
    "occup": "construction",
    "income": 45000,
    "laterality": "right",
    "amai": "B",
    "amai_score": 42,
    "years.begin": 5,
    "drug.age.onset": 20,
    "days.last.use": 7,
    "week.dose": 3,
    "tobc.lastyear": 365,
    "tobc.day": 20,
    "tobc.totyears": 10,
    "illness": "depression",
    "medication": "cocaine"
}

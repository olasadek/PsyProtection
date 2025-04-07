from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

@app.post("/upload-data/")
async def upload_data(
    patient_name: str = Form(...),
    medicine: str = Form(...),
    illness: str = Form(...),
    mri: UploadFile = File(...),
    ehr: UploadFile = File(...)
):
    # Run your multimodal model
    prediction = multimodal_model(mri.file, ehr.file)
    risk_score = prediction["risk_score"]

    # Save info for later use
    save_patient_to_db(patient_name, medicine, illness, risk_score)

    return {
        "alert": risk_score > 0.7,
        "risk": risk_score
    }

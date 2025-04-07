from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

app = FastAPI(title="Drug Abuse Risk Prediction API",
             description="Multimodal model combining neuroimaging and clinical data",
             version="1.0.0")

# Pydantic model for request validation
class PatientData(BaseModel):
    participant_id: str
    age: int
    sex: int  # 1=male, 2=female
    educ_yr: int
    income: Optional[float] = None
    illness: str
    medication: str
    years_medicine_use: float = Form(..., description="Maps to coc.age.onset in model")
    years_begin: Optional[float] = None
    days_last_use: Optional[float] = None
    week_dose: Optional[float] = None

@app.post("/predict-risk/")
async def predict_risk(
    # Demographic/Clinical Data
    participant_id: str = Form(...),
    age: int = Form(...),
    sex: int = Form(...),
    educ_yr: int = Form(...),
    income: float = Form(None),
    illness: str = Form(...),
    medication: str = Form(...),
    years_medicine_use: float = Form(...),
    years_begin: float = Form(None),
    days_last_use: float = Form(None),
    week_dose: float = Form(None),
    
    # File Uploads
    mri_scan: UploadFile = File(...),
    ehr_pdf: Optional[UploadFile] = File(None)
):
    """
    Endpoint for comprehensive risk assessment requiring:
    - Demographic information
    - Clinical history
    - Substance use patterns
    - Neuroimaging data
    - Optional EHR documents
    """
    try:
        # 1. Save uploaded files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mri_path = f"uploads/{participant_id}_mri_{timestamp}.nii.gz"
        
        os.makedirs("uploads", exist_ok=True)
        with open(mri_path, "wb") as buffer:
            buffer.write(await mri_scan.read())
        
        # 2. Prepare clinical features dictionary
        clinical_data = {
            "participant_id": participant_id,
            "age": age,
            "sex": sex,
            "educ_yr": educ_yr,
            "income": income,
            "illness": illness,
            "medication": medication,
            "coc.age.onset": years_medicine_use,  # Explicit mapping
            "years.begin": years_begin if years_begin else 0,
            "days.last.use": days_last_use if days_last_use else 0,
            "week.dose": week_dose if week_dose else 0
        }
        
        # 3. Process EHR if provided (would need OCR/text extraction logic)
        if ehr_pdf:
            ehr_path = f"uploads/{participant_id}_ehr_{timestamp}.pdf"
            with open(ehr_path, "wb") as buffer:
                buffer.write(await ehr_pdf.read())
            clinical_data.update(extract_ehr_data(ehr_path))  # Implement this function
        
        # 4. Get prediction
        result = predict_risk(
            mri_path=mri_path,
            clinical_data=clinical_data
        )
        
        # 5. Format response with all critical info
        return JSONResponse({
            "patient_info": {
                "id": participant_id,
                "age": age,
                "key_risk_factors": {
                    "years_medicine_use": years_medicine_use,
                    "medication": medication,
                    "illness": illness
                }
            },
            "prediction": {
                "risk_score": result['risk_score'],
                "risk_category": "High" if result['is_high_risk'] else "Low",
                "threshold_used": 0.7,
                "important_features": result['important_features']
            },
            "flags": {
                "requires_followup": result['is_high_risk'],
                "urgent": result['risk_score'] > 0.85
            }
        })
    
    except Exception as e:
        logging.error(f"Prediction error for {participant_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def extract_ehr_data(ehr_path: str) -> dict:
    """Placeholder for EHR processing logic"""
    # Implement PDF/text extraction here
    return {}
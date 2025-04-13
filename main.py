from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from model.predict import predict_mri_and_ehr
import ast

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    EHR_features: str = Form(...)
):
    try:
        contents = await file.read()
        ehr_list = ast.literal_eval(EHR_features)
        result = predict_mri_and_ehr(contents, ehr_list)

        return JSONResponse({
            "prediction": result
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

from flask import Flask, request, jsonify
import requests
import json
import io

app = Flask(__name__)

INTERNAL_URL = "http://127.0.0.1:5000"
RAG_URL = "http://127.0.0.1:8000"

@app.route('/analyze_patient', methods=['POST'])
def analyze_patient():
    if 'file' not in request.files or 'EHR_features' not in request.form:
        return jsonify({"error": "Missing MRI file or EHR features"}), 400

    file = request.files['file']
    ehr_raw = request.form['EHR_features']
    patient_id = request.form.get('patient_id', 'unknown')

    try:
        ehr_dict = json.loads(ehr_raw)
    except json.JSONDecodeError:
        return jsonify({"error": "EHR features must be valid JSON"}), 400

    # Validate MRI
    validate_resp = requests.post(f"{INTERNAL_URL}/validate_mri", files={'file': (file.filename, file)})
    validation_result = validate_resp.json()
    file.stream.seek(0)  # rewind for reuse

    if not validation_result.get("valid", False):
        return jsonify({
            "error": "Invalid MRI file",
            "details": validation_result.get("message", "")
        }), 400

    # Run main prediction
    file.stream.seek(0)
    response = requests.post(
        f"{INTERNAL_URL}/drug_abuse_detector",
        files={'file': (file.filename, file)},
        data={
            'patient_id': patient_id,
            'EHR_features': json.dumps(ehr_dict)
        }
    )

    try:
        result = response.json()
    except Exception:
        return jsonify({"error": "Invalid response from internal predictor"}), 500

    return jsonify({
        "patient_id": patient_id,
        "mri_validation": validation_result,
        "prediction_result": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

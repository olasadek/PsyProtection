from flask import Flask, request, jsonify
import requests
import json
import io

app = Flask(__name__)

INTERNAL_URL = "http://127.0.0.1:5000"
RAG_URL = "http://127.0.0.1:8000"

@app.route('/analyze_patient', methods=['POST'])
def analyze_patient():
    # Check for required files and data
    if 'file' not in request.files or 'EHR_features' not in request.form:
        return jsonify({"error": "Missing MRI file or EHR features"}), 400

    file = request.files['file']
    ehr_raw = request.form['EHR_features']
    patient_id = request.form.get('patient_id', 'unknown')

    # Validate EHR JSON
    try:
        ehr_dict = json.loads(ehr_raw)
    except json.JSONDecodeError:
        return jsonify({"error": "EHR features must be valid JSON"}), 400

    # Rewind file for reuse
    file.stream.seek(0)

    # Run main prediction directly (no separate validation step)
    try:
        response = requests.post(
            f"{INTERNAL_URL}/drug_abuse_detector",
            files={'file': (file.filename, file)},
            data={
                'patient_id': patient_id,
                'EHR_features': json.dumps(ehr_dict)
            }
        )
        response.raise_for_status()  # Raises exception for 4XX/5XX status codes
        result = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to process prediction",
            "details": str(e)
        }), 500
    except ValueError:
        return jsonify({"error": "Invalid response from internal predictor"}), 500

    # Return simplified response without validation data
    return jsonify({
        "patient_id": patient_id,
        "prediction_result": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

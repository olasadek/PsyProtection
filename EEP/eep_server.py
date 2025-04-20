from flask import Flask, request, jsonify, make_response
import json
import io
import requests

app = Flask(__name__)

PREDICTION_API_URL = "http://127.0.0.1:5000/drug_abuse_detector"
RAG_API_URL = "http://127.0.0.1:8000/ask_question"
EXPLANATION_API_URL = "http://127.0.0.1:5001/explain"

@app.route('/analyze_patient', methods=['POST'])
def analyze_patient():
    try:
        # --- Validate incoming data ---
        if 'file' not in request.files:
            return jsonify({"error": "No MRI file uploaded"}), 400
        if 'EHR_features' not in request.form:
            return jsonify({"error": "No EHR data provided"}), 400

        file = request.files['file']
        ehr_raw = request.form['EHR_features']
        patient_id = request.form.get('patient_id', 'unknown')

        try:
            ehr_dict = json.loads(ehr_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid EHR JSON format"}), 400

        # --- Forward to Prediction API on port 5000 ---
        files = {'file': (file.filename, file.stream, file.content_type)}
        data = {
            'EHR_features': json.dumps(ehr_dict),
            'patient_id': patient_id
        }
        pred_response = requests.post(PREDICTION_API_URL, files=files, data=data)
        if pred_response.status_code != 200:
            return jsonify({"error": "Prediction API failed", "details": pred_response.text}), 500

        pred_json = pred_response.json()
        prediction = pred_json.get("prediction", {})
        prediction_class = prediction.get("class", 0)

        # --- If drug abuser, ask RAG ---
        query_answer = None
        if prediction_class == 1:
            medication = ehr_dict.get("medication", "")
            illness = ehr_dict.get("illness", "")
            if medication and illness:
                rag_query = f"newest treatments for {medication} addiction for {illness} patients"
                try:
                    rag_response = requests.post(RAG_API_URL, json={"query": rag_query})
                    query_answer = rag_response.json().get("answer", "No answer found.")
                except Exception as e:
                    query_answer = f"RAG error: {str(e)}"

        return jsonify({
            "prediction": prediction,
            "query_answer": query_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain', methods=['POST'])
def analyze_with_explanation():
    try:
        # --- Validate incoming data ---
        if 'file' not in request.files:
            return jsonify({"error": "No MRI file uploaded"}), 400
        if 'EHR_features' not in request.form:
            return jsonify({"error": "No EHR data provided"}), 400

        file = request.files['file']
        ehr_raw = request.form['EHR_features']
        patient_id = request.form.get('patient_id', 'unknown')

        try:
            ehr_dict = json.loads(ehr_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid EHR JSON format"}), 400

        # Make a copy of the file stream for the explanation API
        file_copy = io.BytesIO(file.read())
        file.seek(0)  # Rewind the original file

        # --- Forward to Prediction API on port 5000 ---
        files = {'file': (file.filename, file.stream, file.content_type)}
        data = {
            'EHR_features': json.dumps(ehr_dict),
            'patient_id': patient_id
        }
        pred_response = requests.post(PREDICTION_API_URL, files=files, data=data)
        if pred_response.status_code != 200:
            return jsonify({"error": "Prediction API failed", "details": pred_response.text}), 500

        pred_json = pred_response.json()
        prediction = pred_json.get("prediction", {})
        prediction_class = prediction.get("class", 0)

        # --- If drug abuser, get explanation from port 5001 ---
        heatmap_response = None
        if prediction_class == 1:
            # Prepare data for explanation API
            files_explanation = {'file': (file.filename, file_copy, file.content_type)}
            data_explanation = {
                'EHR_features': json.dumps(ehr_dict),
                'patient_id': patient_id
            }
            
            # Get heatmap from explanation API
            expl_response = requests.post(EXPLANATION_API_URL, files=files_explanation, data=data_explanation)
            
            if expl_response.status_code == 200:
                # Create response with the raw image data
                response = make_response(expl_response.content)
                response.headers.set('Content-Type', 'image/png')
                return response
            else:
                return jsonify({"error": "Explanation API failed", "details": expl_response.text}), 500

        return jsonify({
            "prediction": prediction,
            "message": "No heatmap generated as patient is not predicted as drug abuser"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

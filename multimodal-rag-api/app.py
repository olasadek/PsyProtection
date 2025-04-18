#%%
import os
import requests
import json
from flask import Flask, request, jsonify

#%%
app = Flask(__name__)
#%%
@app.route("/predict_recommendation", methods=["POST"])
def predict_recommendation():
  try:  
    print('hello')
    # Receive image from the request
    image_data = request.files["file"]
    ehr_raw = request.form.get('EHR_features')
    ehr_data = json.loads(ehr_raw)
    files = {
        "file": (image_data.filename, image_data.read(), image_data.content_type)
    }

    data = {
        "EHR_features": ehr_raw
    }

    # Send the image and data to the face detection service
    drug_abuse_detector_response = requests.post(
        os.getenv("drug_abuse_detector_URL"),
        files=files,
        data=data
    )
    
    # After verdict logic
    suggested_query = None
    if drug_abuse_detector_response.json().get("prediction", {}).get("description", None) == "Drug abuser":
            illness = ehr_data.get("illness", "")
            medication = ehr_data.get("medication", "")
            if illness and medication and illness != "none" and medication != "None":
                suggested_query = f"newest treatments for {medication} addiction for {illness} patients"
            
            else:
                suggested_query = None  # If there's no illness or medication, no suggested query
    query_answer = None
    if suggested_query:
            # Send the suggested query to the service on port 8000
            response = requests.post(os.getenv("RAG_URL"), json={"query": suggested_query})
            if response.status_code == 200:
                query_answer = response.json().get("answer", "No answer found.")
            else:
                query_answer = "Error retrieving the answer from query service."
    return jsonify({
            
            "query_answer": query_answer
        })
  except Exception as e:
        return jsonify({"error": str(e)}), 400
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT"))

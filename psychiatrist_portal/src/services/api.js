const API_BASE = 'http://localhost:9000'; // Your eep_server.py port

export const analyzePatient = async (patientId, ehrData, mriFile) => {
  const formData = new FormData();
  formData.append('patient_id', patientId);
  formData.append('EHR_features', JSON.stringify(ehrData));
  formData.append('file', mriFile);

  const response = await fetch(`${API_BASE}/analyze_patient`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to analyze patient');
  }

  return await response.json();
};

export const getPredictions = async (patientId = null) => {
  const url = patientId 
    ? `${API_BASE}/get_predictions?patient_id=${patientId}`
    : `${API_BASE}/get_predictions`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to fetch predictions');
  }
  return await response.json();
};
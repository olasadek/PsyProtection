const API_BASE = 'http://localhost:9000';

// Call this route to analyze the patient (returns JSON with prediction data)
export const analyzePatient = async (patientId, ehrData, mriFile) => {
  const formData = new FormData();
  formData.append('patient_id', patientId);
  formData.append('EHR_features', JSON.stringify(ehrData));
  formData.append('file', mriFile);

  console.log('Analyzing patient on:', `${API_BASE}/analyze_patient`);
  const response = await fetch(`${API_BASE}/analyze_patient`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Failed to analyze patient: ${errText}`);
  }
  const data = await response.json();
  console.log('Analysis response:', data);
  return data;
};

// Call this route to get explanation (returns image data)
// Notice that here we assume the explanation route is /explain on port 9000.
export const getExplanation = async (patientId, ehrData, mriFile) => {
  const formData = new FormData();
  formData.append('patient_id', patientId);
  formData.append('EHR_features', JSON.stringify(ehrData));
  formData.append('file', mriFile);

  console.log('Requesting explanation on:', `${API_BASE}/explain`);
  const response = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Failed to get explanation: ${errText}`);
  }

  // Since this returns an image, convert it to a blob and then to an object URL.
  const blob = await response.blob();
  const imageUrl = URL.createObjectURL(blob);
  console.log('Explanation image URL:', imageUrl);
  return imageUrl;
};

import React from 'react';
import Dashboard from './components/Dashboard';
import { analyzePatient, getExplanation } from './services/api';

function App() {
  const handleAnalyze = async (ehrData, mriFile) => {
    const patientId = 'patient-123'; // Change or generate as needed

    // Call analyzePatient to get prediction data.
    const analysisResponse = await analyzePatient(patientId, ehrData, mriFile);

    // Prepare the result object.
    let combinedResult = {
      patient_id: patientId,
      prediction: analysisResponse.prediction,
      query_answer: analysisResponse.query_answer || "",
      explanation: ""
    };

    // If predicted as a drug abuser, call getExplanation.
    if (analysisResponse.prediction && analysisResponse.prediction.class === 1) {
      console.log('Drug abuser detected. Fetching explanation...');
      const explanationUrl = await getExplanation(patientId, ehrData, mriFile);
      combinedResult.explanation = explanationUrl;
    } else {
      console.log('Patient not predicted as drug abuser; skipping explanation.');
    }

    return combinedResult;
  };

  return (
    <div className="App">
      <h1 style={{ textAlign: "center", margin: "20px 0", color: "#fff" }}>
        Psychiatric Patient Analyzer
      </h1>
      <Dashboard onAnalyze={handleAnalyze} />
    </div>
  );
}

export default App;

import { useState } from 'react';
import { analyzePatient } from '../services/api';
import PatientForm from './PatientForm';
import MRIUpload from './MRIUpload';
import PredictionResult from './PredictionResult';

const Dashboard = () => {
  const [patientId, setPatientId] = useState('');
  const [ehrData, setEhrData] = useState(null);
  const [mriFile, setMriFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleSubmitEHR = (data) => {
    setEhrData(data);
    // Generate a patient ID if not provided
    if (!patientId) {
      setPatientId(`PAT-${Date.now().toString().slice(-6)}`);
    }
  };

  const handleAnalyze = async () => {
    if (!ehrData || !mriFile) return;
    
    setIsAnalyzing(true);
    try {
      const analysisResult = await analyzePatient(patientId, ehrData, mriFile);
      setResult(analysisResult);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Failed to analyze patient');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="dashboard">
      <h1>New Patient Analysis</h1>
      
      <div className="patient-id-input">
        <label>Patient ID:</label>
        <input 
          type="text" 
          value={patientId} 
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="Leave blank to generate automatically"
        />
      </div>
      
      <div className="analysis-steps">
        <div className="step">
          <h2>Step 1: Enter Patient Data</h2>
          <PatientForm onSubmit={handleSubmitEHR} />
        </div>
        
        <div className="step">
          <h2>Step 2: Upload MRI Scan</h2>
          <MRIUpload 
            onUpload={setMriFile} 
            patientId={patientId}
          />
        </div>
        
        {ehrData && mriFile && (
          <div className="step">
            <h2>Step 3: Run Analysis</h2>
            <button 
              onClick={handleAnalyze} 
              disabled={isAnalyzing}
              className="analyze-btn"
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Patient'}
            </button>
          </div>
        )}
      </div>
      
      {result && <PredictionResult result={result} />}
    </div>
  );
};

export default Dashboard;
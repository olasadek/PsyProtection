import React, { useState } from 'react';
import PatientForm from './PatientForm';
import PredictionResult from './PredictionResult';
import Loading from './Loading';
import './Dashboard.css';

const Dashboard = ({ onAnalyze }) => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFormSubmit = async (ehrData, mriFile) => {
    setLoading(true);
    try {
      const response = await onAnalyze(ehrData, mriFile);
      setResult(response);
    } catch (error) {
      console.error("Error analyzing patient:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <aside className="dashboard-sidebar">
        <h2>Patient Details</h2>
        <PatientForm onSubmit={handleFormSubmit} />
      </aside>
      <main className="dashboard-content">
        <h2>Analysis Result</h2>
        {loading ? (
          <Loading />
        ) : (
          result ? (
            <PredictionResult result={result} />
          ) : (
            <p>Please submit patient data to get started.</p>
          )
        )}
      </main>
    </div>
  );
};

export default Dashboard;

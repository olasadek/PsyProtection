import { useEffect, useState } from 'react';
import { getPredictions } from '../services/api';

const PatientList = ({ onSelectPatient }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const data = await getPredictions();
        setPatients(data);
      } catch (error) {
        console.error('Error fetching patients:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchPatients();
  }, []);

  if (loading) return <div>Loading patients...</div>;

  return (
    <div className="patient-list">
      <h2>Recent Patients</h2>
      <div className="list-container">
        {patients.length === 0 ? (
          <p>No patients found</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Date</th>
                <th>Verdict</th>
                <th>Confidence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(patient => (
                <tr key={patient.id}>
                  <td>{patient.patient_id}</td>
                  <td>{new Date(patient.timestamp).toLocaleString()}</td>
                  <td className={patient.prediction_class ? 'positive' : 'negative'}>
                    {patient.verdict}
                  </td>
                  <td>{(patient.prediction_prob * 100).toFixed(2)}%</td>
                  <td>
                    <button onClick={() => onSelectPatient(patient)}>View Details</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default PatientList;
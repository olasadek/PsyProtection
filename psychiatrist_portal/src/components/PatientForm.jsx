import { useState } from 'react';

const PatientForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    age: '',
    "years of education": '',
    "occupation level": '',
    "annual income": '',
    "handedness": '',
    "AMAI category": '',
    "AMAI score": '',
    "years since first drug use": '',
    "age of drug onset": '',
    "days since last drug use": '',
    "weekly dose": '',
    "tobacco consumed since last year": '',
    "tobacco per day": '',
    "years of tobacco use": '',
    "current illness": '',
    "current medication": ''
  });

  const [mriFile, setMriFile] = useState(null);

  const keyMap = {
    "age": "age",
    "years of education": "educ_yr",
    "occupation level": "occup",
    "annual income": "income",
    "handedness": "laterality",
    "AMAI category": "amai",
    "AMAI score": "amai_score",
    "years since first drug use": "years.begin",
    "age of drug onset": "drug.age.onset",
    "days since last drug use": "days.last.use",
    "weekly dose": "week.dose",
    "tobacco consumed since last year": "tobc.lastyear",
    "tobacco per day": "tobc.day",
    "years of tobacco use": "tobc.totyears",
    "current illness": "illness",
    "current medication": "medication"
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Normalize keys before sending
    const transformedData = {};
    for (const [key, value] of Object.entries(formData)) {
      const mappedKey = keyMap[key] || key;
      transformedData[mappedKey] = value;
    }

    if (!mriFile) {
      alert("Please upload an MRI file.");
      return;
    }

    try {
      await onSubmit(transformedData, mriFile);
    } catch (error) {
      console.error("Error submitting form:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="patient-form">
      <h2>Patient EHR Data</h2>
      <div className="form-grid">
        {Object.keys(formData).map((field) => (
          <div className="form-group" key={field}>
            <label>{field}</label>
            <input
              type="text"
              name={field}
              value={formData[field]}
              onChange={handleChange}
              required={field === 'age'}
            />
          </div>
        ))}
        <div className="form-group">
          <label>MRI File</label>
          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={(e) => setMriFile(e.target.files[0])}
            required
          />
        </div>
      </div>
      <button type="submit" className="submit-btn">Submit</button>
    </form>
  );
};

export default PatientForm;

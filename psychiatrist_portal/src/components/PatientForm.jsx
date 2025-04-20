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

  const [heatmapUrl, setHeatmapUrl] = useState(null);

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
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Normalize key names before sending to backend
    const transformedData = {};
    for (const [key, value] of Object.entries(formData)) {
      const mappedKey = keyMap[key] || key; // Mapping the form key to model key
      transformedData[mappedKey] = value;
    }

    try {
      // Call the onSubmit function and await its response.
      const responseData = await onSubmit(transformedData);
      // If the response includes an explanation (heatmap URL), set it.
      if (responseData && responseData.explanation) {
        setHeatmapUrl(`http://127.0.0.1:5000${responseData.explanation}`);
      }
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
              required={field === 'age'} // age is required, others optional
            />
          </div>
        ))}
      </div>
      <button type="submit" className="submit-btn">Save EHR Data</button>

      {heatmapUrl && (
        <div className="heatmap-container">
          <h3>MRI Heatmap Explanation</h3>
          <img src={heatmapUrl} alt="Occlusion Heatmap" className="heatmap-img" />
        </div>
      )}
    </form>
  );
};

export default PatientForm;
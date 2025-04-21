const PredictionResult = ({ result }) => {
  // Log the result so we can see if we have data.
  console.log('PredictionResult received:', result);
  
  if (!result || !result.prediction) return null;

  const prediction = result.prediction || {};
  const explanation = result.explanation || '';
  const queryAnswer = result.query_answer || '';

  return (
    <div className="prediction-result">
      <h2>Analysis Results</h2>

      <div className="result-card">
        <h3>Patient ID: {result.patient_id || 'N/A'}</h3>

        <div className="result-section">
          <h4>Prediction</h4>
          <div className={`verdict ${prediction.class ? 'positive' : 'negative'}`}>
            {prediction.description || 'No prediction available'}
          </div>
          <div className="confidence">
            Confidence: {prediction.probability ? (prediction.probability * 100).toFixed(2) + '%' : 'N/A'}
          </div>
        </div>

        {explanation && (
          <div className="result-section">
            <h4>Explanation</h4>
            <img
              src={explanation}
              alt="MRI heatmap explanation"
              className="heatmap"
            />
          </div>
        )}

        {queryAnswer && (
          <div className="result-section">
            <h4>Treatment Recommendation</h4>
            <p className="treatment-answer">{queryAnswer}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionResult;

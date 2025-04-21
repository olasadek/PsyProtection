const PredictionResult = ({ result }) => {
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
          <div className="result-section explanation-section">
            <h4>Model Explanation</h4>
            <div className="heatmap-container" style={{ textAlign: 'center' }}>
              <img
                src={explanation}
                alt="MRI heatmap explanation"
                className="heatmap"
                style={{ margin: '0 auto' }}
              />
            </div>
            <p className="heatmap-caption" style={{ 
              fontSize: '1.2rem', 
              textAlign: 'center',
              margin: '20px auto',
              maxWidth: '800px'
            }}>
              The heatmap highlights regions of your MRI scan that most influenced 
              our AI model's prediction. Warmer colors (red/yellow) indicate areas 
              of greater significance in the analysis.
              Abnormalities in: the basal ganglia, the extended amygdala, 
              and the prefrontal cortex.
            </p>
          </div>
        )}

        {queryAnswer && (
          <div className="result-section treatment-section" style={{ textAlign: 'center' }}>
            <h4>Treatment Recommendation</h4>
            <p className="treatment-answer" style={{
              fontSize: '1.2rem',
              fontWeight: 'bold',
              margin: '20px auto',
              maxWidth: '800px'
            }}>
              {queryAnswer}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionResult;


import { useState } from 'react';
import './styles.css';

const API_URL =
  'https://ai-based-lung-cancer-detection-1.onrender.com';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  function handleFileChange(event) {
    const selected = event.target.files?.[0];

    if (!selected) return;

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) {
      setError('Please select a JPG, PNG, or WebP image.');
      return;
    }

    if (selected.size > 15 * 1024 * 1024) {
      setError('Please select an image smaller than 15 MB.');
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError('');
  }

  async function analyzeImage() {
    if (!file || loading) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const responseText = await response.text();
      let data;

      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error(
          `The server returned an invalid response (HTTP ${response.status}).`
        );
      }

      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          `Server error: HTTP ${response.status}`
        );
      }

      const predictedClass =
        data.predicted_class ||
        data.prediction ||
        data.class_name ||
        data.predicted_label;

      if (!predictedClass) {
        throw new Error(
          'The API responded, but no predicted class was found. Check the response format in backend/main.py.'
        );
      }

      setResult({
        ...data,
        predicted_class: predictedClass,
      });
    } catch (err) {
      console.error('Lung cancer API error:', err);

      if (err instanceof TypeError) {
        setError(
          'Could not connect to the API. Check your internet connection, Render service status, and backend CORS settings.'
        );
      } else {
        setError(err.message || 'An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  }

  function resetApp() {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview('');
    setResult(null);
    setError('');
  }

  const confidence = Number(
    result?.confidence_percent ??
    (result?.confidence != null
      ? Number(result.confidence) * 100
      : 0)
  );

  const scores = result?.scores || {};

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">+</div>
          <div>
            <h2>Lung Cancer Detection</h2>
            <p>AI-Based Medical Image Research</p>
          </div>
        </div>

        <span className="connection-status">
          Research application
        </span>
      </header>

      <main className="main-content">
        <section className="welcome">
          <p className="eyebrow">MEDICAL IMAGE ANALYSIS</p>
          <h1>CT Image Classification</h1>
          <p>
            Upload an image to test your trained three-class
            machine-learning model.
          </p>
        </section>

        <section className="content-grid">
          <div className="panel">
            <h2>Upload Image</h2>
            <p className="muted">
              Supported formats: JPG, PNG and WebP.
            </p>

            <label className="upload-box">
              {preview ? (
                <img
                  className="scan-preview"
                  src={preview}
                  alt="Selected image preview"
                />
              ) : (
                <div className="upload-placeholder">
                  <span className="upload-symbol">↑</span>
                  <strong>Select a CT image</strong>
                  <span>Click here to browse your files</span>
                </div>
              )}

              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
              />
            </label>

            {file && (
              <p className="file-name">
                Selected: {file.name}
              </p>
            )}

            <div className="button-row">
              <button
                className="button secondary"
                type="button"
                onClick={resetApp}
                disabled={loading}
              >
                Clear
              </button>

              <button
                className="button primary"
                type="button"
                onClick={analyzeImage}
                disabled={!file || loading}
              >
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </button>
            </div>

            {loading && (
              <p className="status-message">
                Connecting to the model server. The first request
                may take longer while Render starts the service.
              </p>
            )}

            {error && (
              <div className="error-message" role="alert">
                <strong>Analysis failed</strong>
                <p>{error}</p>
                <button
                  className="button secondary"
                  type="button"
                  onClick={analyzeImage}
                  disabled={!file || loading}
                >
                  Try Again
                </button>
              </div>
            )}
          </div>

          <div className="panel results-panel">
            <h2>Analysis Results</h2>
            <p className="muted">
              The result appears here after a successful API response.
            </p>

            {!result && !loading && (
              <div className="empty-results">
                <div className="result-symbol">i</div>
                <h3>Waiting for image analysis</h3>
                <p>
                  Select an image and click Run Analysis to
                  request a prediction from your model.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-results">
                <h3>Processing image...</h3>
                <p>Please wait for the server response.</p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <p className="result-label">
                  PREDICTED CLASS
                </p>

                <h3 className="prediction">
                  {result.predicted_class}
                </h3>

                <p className="muted">
                  Model output only. This is not a medical diagnosis.
                </p>

                <div className="confidence-section">
                  <div className="confidence-heading">
                    <span>Reported confidence</span>
                    <strong>
                      {Number.isFinite(confidence)
                        ? `${Math.max(0, Math.min(100, confidence)).toFixed(2)}%`
                        : 'Not provided'}
                    </strong>
                  </div>

                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${
                          Number.isFinite(confidence)
                            ? Math.max(0, Math.min(100, confidence))
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>

                {Object.keys(scores).length > 0 && (
                  <div className="score-list">
                    <h4>Class scores returned by the API</h4>

                    {Object.entries(scores).map(([name, value]) => {
                      const numericValue = Number(value);
                      const percentage =
                        numericValue <= 1 && numericValue >= 0
                          ? numericValue * 100
                          : numericValue;

                      const safePercentage = Number.isFinite(percentage)
                        ? Math.max(0, Math.min(100, percentage))
                        : 0;

                      return (
                        <div className="score-item" key={name}>
                          <div className="confidence-heading">
                            <span>{name}</span>
                            <strong>{safePercentage.toFixed(2)}%</strong>
                          </div>
                          <div className="progress-track">
                            <div
                              className="progress-fill"
                              style={{ width: `${safePercentage}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                <div className="disclaimer">
                  Educational research only. This model has not
                  been validated for clinical use. Do not use its
                  predictions to make health decisions.
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="information-panel">
          <h2>Project Workflow</h2>
          <div className="workflow-grid">
            <div>
              <strong>01. Upload</strong>
              <p>Select a supported image file.</p>
            </div>
            <div>
              <strong>02. Predict</strong>
              <p>Send the image to the Render API.</p>
            </div>
            <div>
              <strong>03. Review</strong>
              <p>Inspect the returned class and scores.</p>
            </div>
          </div>
        </section>

        <footer className="app-footer">
          AI-Based Lung Cancer Detection · Student Research Project
          <p>
            Not a diagnostic tool. Clinical decisions require
            qualified medical professionals.
          </p>
        </footer>
      </main>
    </div>
  );
}
